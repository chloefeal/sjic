"""RTSP 拉流：后台持续 grab，避免推理阻塞导致 UDP 丢包/重连。"""
import threading
import time

import cv2


def open_capture(url):
    if not url:
        return None
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def _open_for_reconnect(url):
    """重连时尽量走与主路径相同的 RtspCapture，失败再退回 raw VideoCapture。"""
    try:
        from utils.stream import open_rtsp_capture
        return open_rtsp_capture(url)
    except Exception:
        return open_capture(url)


def is_valid_frame(frame):
    if frame is None:
        return False
    try:
        return frame.size > 0 and len(frame.shape) >= 2 and frame.shape[0] >= 16
    except Exception:
        return False


class FramePump:
    """独立线程读流，主循环只取最新帧。

    为何要 copy：
    OpenCV retrieve() 常复用内部缓冲，下一次 retrieve 会覆盖同一块内存。
    在 _store 里立刻 copy 成自有缓冲；解码已限流，比「消费时再 copy」更不容易花屏。
    """

    def __init__(self, cap, stop_event, url, logger, open_fn=None, max_decode_fps=12.0):
        self.cap = cap
        self.stop_event = stop_event
        self.url = url
        self.logger = logger
        self.open_fn = open_fn or _open_for_reconnect
        # 解码上限：推理通常远慢于摄像头帧率，全速 decode 只会空烧 CPU
        self.max_decode_fps = float(max_decode_fps) if max_decode_fps else 0.0
        self._lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._fail = 0
        self._released = False
        self._last_decode_t = 0.0
        self._decode_n = 0
        self._consume_n = 0
        self._stats_t = time.time()
        self._thread = threading.Thread(target=self._loop, name='rtsp-pump', daemon=True)

    def start(self):
        self._thread.start()

    def _store(self, frame):
        # retrieve 返回后、下次 grab/retrieve 前 copy，避免与 OpenCV 内部缓冲共享
        owned = frame.copy()
        with self._lock:
            self._frame = owned
            self._seq += 1
            self._fail = 0
            self._decode_n += 1

    def _maybe_log_stats(self):
        now = time.time()
        dt = now - self._stats_t
        if dt < 5.0:
            return
        decode_fps = self._decode_n / dt
        consume_fps = self._consume_n / dt
        self.logger.info(
            f"RTSP pump: decode={decode_fps:.1f}fps consume={consume_fps:.1f}fps "
            f"(drop≈{max(0.0, decode_fps - consume_fps):.1f}fps)"
        )
        self._decode_n = 0
        self._consume_n = 0
        self._stats_t = now

    def _read_one(self):
        """单次 grab+retrieve。不要连 grab 多次：每次 grab 都会阻塞等下一帧，
        连抓 N 次会把解码上限压到 摄像头fps/N（日志里 ~2fps 就是这个原因）。
        积压排空靠限流窗口里的 _grab_only。
        """
        with self._cap_lock:
            if self.stop_event.is_set() or self.cap is None:
                return False, None
            if not self.cap.grab():
                return False, None
            return self.cap.retrieve()

    def _grab_only(self):
        """只 grab 不 decode，用来在限流间隔内排空网络缓冲。"""
        with self._cap_lock:
            if self.stop_event.is_set() or self.cap is None:
                return False
            return bool(self.cap.grab())

    def _take_cap(self):
        with self._cap_lock:
            cap = self.cap
            self.cap = None
            return cap

    def _close_cap(self, cap):
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            pass

    def _reconnect(self):
        if self.stop_event.is_set():
            return
        self.logger.warning("RTSP pump reconnecting...")
        old = self._take_cap()
        self._close_cap(old)
        for _ in range(20):
            if self.stop_event.is_set():
                return
            time.sleep(0.1)
        if self.stop_event.is_set():
            return
        try:
            cap = self.open_fn(self.url)
        except Exception as e:
            self.logger.error(f"RTSP pump reconnect failed: {e}")
            time.sleep(1)
            return
        if cap is None or not cap.isOpened():
            self.logger.error("RTSP pump reconnect failed")
            self._close_cap(cap)
            time.sleep(1)
            return
        with self._cap_lock:
            if self.stop_event.is_set():
                self._close_cap(cap)
                return
            self.cap = cap
            self._fail = 0
        self.logger.info("RTSP pump reconnected")

    def _loop(self):
        min_interval = (1.0 / self.max_decode_fps) if self.max_decode_fps > 0 else 0.0
        while not self.stop_event.is_set():
            try:
                now = time.time()
                if min_interval and (now - self._last_decode_t) < min_interval:
                    # 限流窗口内只 grab 排空，避免缓冲堆积导致延时/重连
                    if not self._grab_only():
                        self._fail += 1
                        if self._fail >= 30:
                            self._reconnect()
                        else:
                            time.sleep(0.01)
                    else:
                        self._fail = 0
                    self._maybe_log_stats()
                    continue

                ret, frame = self._read_one()
                if self.stop_event.is_set():
                    break
                if not ret:
                    self._fail += 1
                    if self._fail >= 30:
                        self._reconnect()
                    else:
                        time.sleep(0.03)
                    continue
                self._last_decode_t = time.time()
                if is_valid_frame(frame):
                    self._store(frame)
                else:
                    self._fail += 1
                self._maybe_log_stats()
            except Exception as e:
                if self.stop_event.is_set():
                    break
                self.logger.warning(f"RTSP pump error: {e}")
                self._fail += 1
                time.sleep(0.05)

    def get_latest(self, last_seq=0, wait_sec=1.0):
        """返回 (frame, seq)。frame 已在 _store 时 copy，可安全给推理用。无新帧时 frame 为 None。"""
        deadline = time.time() + wait_sec
        while not self.stop_event.is_set():
            with self._lock:
                if self._seq > last_seq and self._frame is not None:
                    out = self._frame
                    seq = self._seq
                    self._consume_n += 1
                    return out, seq
            if time.time() >= deadline:
                return None, last_seq
            time.sleep(0.01)
        return None, last_seq

    def release(self):
        """先等泵线程离开 OpenCV native 调用，再释放 VideoCapture，避免 heap corruption。"""
        if self._released:
            return
        self._released = True
        self.stop_event.set()
        thread = self._thread
        if thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=15.0)
            if thread.is_alive():
                self.logger.warning(
                    "RTSP pump still running after stop; waiting for in-flight grab before release"
                )
        cap = self._take_cap()
        self._close_cap(cap)
        with self._lock:
            self._frame = None
