import ctypes
import logging
import os
import subprocess
import threading
import time

import cv2

logger = logging.getLogger('utils.stream')

# 只保留 TCP。多余 option 在部分 OpenCV/FFmpeg 上会让整串参数失效，回退成 UDP。
_FFMPEG_TCP_OPTIONS = "rtsp_transport;tcp"
_MAX_SKIP_FRAMES = 15
_GST_ELEM_CACHE = {}
_GST_SUCCESS = {}  # rtsp_url -> (pipeline, label)
_DECODE_PROBE_LOGGED = False


def _silence_libav_logs():
    """压掉 FFmpeg 打到 stderr 的 H.264 MB / RTP 解码噪声。"""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except Exception:
        pass

    for name in (
        getattr(cv2, "__file__", None),
        "libavutil.so.59",
        "libavutil.so.58",
        "libavutil.so.57",
        "libavutil.so.56",
        "libavutil.so",
    ):
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
            lib.av_log_set_level(8)  # AV_LOG_FATAL，低于 ERROR
        except Exception:
            continue


def _configure_ffmpeg_tcp():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _FFMPEG_TCP_OPTIONS
    os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "65536")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def _hwdec_wanted():
    v = os.environ.get("SJIC_RTSP_HWDEC", "1").strip().lower()
    return v not in ("0", "false", "off", "cpu", "no")


def _gst_has_element(name):
    cached = _GST_ELEM_CACHE.get(name)
    if cached is not None:
        return cached
    ok = False
    try:
        r = subprocess.run(
            ["gst-inspect-1.0", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        ok = r.returncode == 0
    except Exception:
        ok = False
    _GST_ELEM_CACHE[name] = ok
    return ok


def _nvdec_kind():
    """jetson / nvidia / ''。看设备节点，不依赖 config.yaml 是否填对。"""
    if os.path.exists("/dev/nvhost-nvdec") or os.path.exists("/dev/nvhost-ctrl-gpu"):
        return "jetson"
    if os.path.exists("/dev/nvidia0"):
        return "nvidia"
    return ""


def _gst_src_and_tail(rtsp_url):
    location = rtsp_url.replace("\\", "\\\\").replace('"', '\\"')
    src = (
        f'rtspsrc location="{location}" protocols=tcp latency=200 retry=5 '
        f'timeout=5000000 tcp-timeout=5000000 drop-on-latency=true'
    )
    # leaky queue 放在解码器前面：推理期间 Python 不 read 时解码器堵住，
    # 旧压缩包在这里丢掉，只保留最新一包。appsink 不 drop，避免空转解码。
    q = "queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0"
    sink = "appsink max-buffers=1 drop=false sync=false"
    bgr = f"videoconvert ! video/x-raw,format=BGR ! {sink}"
    return src, q, bgr


def _log_decode_probe_once(hw, kind, jetson, desktop, rk):
    global _DECODE_PROBE_LOGGED
    if _DECODE_PROBE_LOGGED:
        return
    _DECODE_PROBE_LOGGED = True
    elems = [
        name for name in (
            "nvv4l2decoder", "nvvidconv", "nvh264dec", "nvh265dec", "mppvideodec",
        ) if _gst_has_element(name)
    ]
    logger.info(
        "RTSP decode probe: hwdec=%s nvdec=%s jetson=%s desktop=%s mpp=%s gst_hw=%s",
        "on" if hw else "off",
        kind or "none",
        jetson,
        desktop,
        rk,
        elems or "none",
    )


def _gst_candidates(rtsp_url):
    """按优先级列出 GStreamer 管道：NVDEC/MPP 硬解 → CPU 软解。"""
    src, q, bgr = _gst_src_and_tail(rtsp_url)
    out = []
    hw = _hwdec_wanted()
    kind = _nvdec_kind() if hw else ""
    inspect_nvv4l2 = _gst_has_element("nvv4l2decoder")
    inspect_nvh264 = _gst_has_element("nvh264dec")
    inspect_nvh265 = _gst_has_element("nvh265dec")
    inspect_mpp = _gst_has_element("mppvideodec")
    rk_dev = os.path.exists("/dev/mpp_service") or os.path.exists("/dev/rkvdec")

    jetson = hw and (kind == "jetson" or inspect_nvv4l2)
    desktop = hw and kind != "jetson" and (kind == "nvidia" or inspect_nvh264 or inspect_nvh265)
    rk = hw and (inspect_mpp or rk_dev)
    _log_decode_probe_once(hw, kind, jetson, desktop, rk)

    if jetson:
        # NVMM → BGRx → BGR，OpenCV appsink 只要系统内存 BGR。
        nv2bgr = f"{q} ! nvvidconv ! video/x-raw,format=BGRx ! {bgr}"
        # decodebin 一次连接即可适配 H.264/H.265，L4T 通常会插 nvv4l2decoder（NVDEC）。
        out.append((
            f"{src} ! decodebin ! {nv2bgr}",
            "gstreamer-nvdec",
        ))
        out.append((
            f"{src} ! rtph264depay ! h264parse ! nvv4l2decoder ! {nv2bgr}",
            "gstreamer-nvdec-h264",
        ))
        out.append((
            f"{src} ! rtph265depay ! h265parse ! nvv4l2decoder ! {nv2bgr}",
            "gstreamer-nvdec-h265",
        ))
    elif desktop:
        out.append((
            f"{src} ! rtph264depay ! h264parse ! nvh264dec ! {q} ! {bgr}",
            "gstreamer-nvdec-h264",
        ))
        if inspect_nvh265 or kind == "nvidia":
            out.append((
                f"{src} ! rtph265depay ! h265parse ! nvh265dec ! {q} ! {bgr}",
                "gstreamer-nvdec-h265",
            ))
    elif rk:
        out.append((
            f"{src} ! rtph264depay ! h264parse ! mppvideodec ! {q} ! {bgr}",
            "gstreamer-mpp-h264",
        ))
        out.append((
            f"{src} ! rtph265depay ! h265parse ! mppvideodec ! {q} ! {bgr}",
            "gstreamer-mpp-h265",
        ))

    out.append((
        f"{src} ! rtph264depay ! h264parse ! {q} ! avdec_h264 ! {bgr}",
        "gstreamer-cpu-h264",
    ))
    out.append((
        f"{src} ! rtph265depay ! h265parse ! {q} ! avdec_h265 ! {bgr}",
        "gstreamer-cpu-h265",
    ))
    return out


def _try_gst(pipeline, label):
    if not hasattr(cv2, "CAP_GSTREAMER"):
        return None
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        logger.info("Opened RTSP via %s (drop-old-before-decode)", label)
        return _tune_cap(cap)
    cap.release()
    logger.info("%s unavailable, trying next backend", label)
    return None


def _tune_cap(cap):
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
    except Exception:
        pass
    return cap


def _open_raw(rtsp_url):
    """GStreamer 硬解优先（NVDEC/MPP）→ CPU 软解 → FFmpeg-TCP → 默认。返回 (cap, backend)。"""
    candidates = _gst_candidates(rtsp_url)
    cached = _GST_SUCCESS.get(rtsp_url)
    if cached:
        candidates = [cached] + [c for c in candidates if c != cached]

    for pipeline, label in candidates:
        cap = _try_gst(pipeline, label)
        if cap is not None:
            _GST_SUCCESS[rtsp_url] = (pipeline, label)
            return cap, label

    logger.info("GStreamer TCP unavailable, trying FFmpeg TCP")
    _configure_ffmpeg_tcp()
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        hw_prop = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
        hw_any = getattr(cv2, "VIDEO_ACCELERATION_ANY", None)
        if hw_prop is not None and hw_any is not None:
            try:
                cap.set(hw_prop, hw_any)
            except Exception:
                pass
        logger.info("Opened RTSP via FFmpeg TCP")
        return _tune_cap(cap), "ffmpeg"
    cap.release()

    logger.warning("FFmpeg TCP open failed, falling back to default (likely UDP): %s", rtsp_url)
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        return _tune_cap(cap), "ffmpeg"
    cap.release()
    return None, None


class RtspCapture:
    """
    推理线程按需取最新帧，没有后台解码线程。

    GStreamer：解码器前 leaky queue，不 read 时只收包、丢旧压缩帧，CPU 接近空闲。
    FFmpeg：TCP 把包堆在内核；下次 grab 按间隔跳过积压帧，再 retrieve 当前这一帧。
    """

    def __init__(self, rtsp_url):
        self._url = rtsp_url
        self._cap = None
        self._backend = "ffmpeg"
        self._fps = 25.0
        self._last_grab = 0.0
        self._lock = threading.Lock()

    def start(self):
        _silence_libav_logs()
        cap, backend = _open_raw(self._url)
        if cap is None:
            raise ConnectionError(f"Cannot open camera stream at {self._url}")
        self._cap = cap
        self._backend = backend
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 1 else 25.0
        self._last_grab = 0.0
        logger.info("RTSP ready backend=%s fps=%.1f", backend, self._fps)
        return self

    def _discard_stale(self):
        """丢掉推理期间积压的旧帧，只解/保留最新一帧。GStreamer 管道已在压缩域丢弃。"""
        if self._cap is None or str(self._backend).startswith("gstreamer"):
            return
        if self._last_grab <= 0:
            return
        elapsed = time.time() - self._last_grab
        skip = int(elapsed * self._fps) - 1
        skip = min(max(skip, 0), _MAX_SKIP_FRAMES)
        for _ in range(skip):
            if not self._cap.grab():
                break

    def grab(self):
        with self._lock:
            if self._cap is None:
                return False
            self._discard_stale()
            ok = self._cap.grab()
            if ok:
                self._last_grab = time.time()
            return ok

    def retrieve(self):
        with self._lock:
            if self._cap is None:
                return False, None
            return self._cap.retrieve()

    def read(self):
        with self._lock:
            if self._cap is None:
                return False, None
            self._discard_stale()
            if not self._cap.grab():
                return False, None
            self._last_grab = time.time()
            return self._cap.retrieve()

    def get(self, prop):
        with self._lock:
            if self._cap is None:
                return 0
            return self._cap.get(prop)

    def set(self, prop, value):
        with self._lock:
            if self._cap is None:
                return False
            return self._cap.set(prop, value)

    def isOpened(self):
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def release(self):
        with self._lock:
            cap = self._cap
            self._cap = None
            if cap is None:
                return
            try:
                cap.release()
            except Exception:
                pass


def open_rtsp_capture(rtsp_url, buffer_size=1, warmup_grabs=5):
    """打开 RTSP。grab/read 时跳到最新帧，无后台读流线程。"""
    if not rtsp_url:
        raise ValueError("rtsp_url is empty")
    _ = buffer_size, warmup_grabs
    return RtspCapture(rtsp_url).start()


_silence_libav_logs()
