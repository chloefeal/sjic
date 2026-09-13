"""
姿态行为引擎：张望 / 低头 / 不看屏幕 / 手托下巴 / 倒地 / 手指屏幕 /
捂嘴报题 / AI 智能眼镜 / 考试员不巡场。

依赖姿态模型（YOLO-Pose 等）经 runtime.infer 返回 keypoints。
若当前 runtime 仅返回 boxes，则退化为「有人框 + 宽高比/位置」粗判，并在日志中提示。
"""
from .base import BaseAlgorithm
import math
import time
from datetime import datetime
from utils.calc import get_letterbox_params, preprocess
from utils.rtsp import FramePump, is_valid_frame
from utils.schedule import is_item_active_now


# COCO 17 keypoints 索引
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def _kp_ok(kps, idx, min_conf=0.25):
    if kps is None or idx >= len(kps):
        return False
    x, y, c = kps[idx]
    return c >= min_conf and x > 0 and y > 0


def _kp(kps, idx):
    return kps[idx][0], kps[idx][1]


def _shoulder_width(kps, default=80.0):
    if _kp_ok(kps, L_SHOULDER) and _kp_ok(kps, R_SHOULDER):
        return max(abs(_kp(kps, R_SHOULDER)[0] - _kp(kps, L_SHOULDER)[0]), 1.0)
    return default


def estimate_yaw_pitch(kps):
    """粗略 yaw/pitch（度）。正 yaw=左转脸，正 pitch=低头倾向。"""
    if not (_kp_ok(kps, L_SHOULDER) and _kp_ok(kps, R_SHOULDER) and _kp_ok(kps, NOSE)):
        return None, None
    ls = _kp(kps, L_SHOULDER)
    rs = _kp(kps, R_SHOULDER)
    nose = _kp(kps, NOSE)
    mid_x = (ls[0] + rs[0]) / 2.0
    mid_y = (ls[1] + rs[1]) / 2.0
    shoulder_w = max(abs(rs[0] - ls[0]), 1.0)
    yaw = max(-90.0, min(90.0, ((nose[0] - mid_x) / shoulder_w) * 90.0))
    pitch = max(-90.0, min(90.0, ((nose[1] - mid_y) / shoulder_w) * 90.0))
    return yaw, pitch


def is_looking_aside(kps, yaw_degrees=45):
    yaw, _ = estimate_yaw_pitch(kps)
    if yaw is None:
        return False
    return abs(yaw) >= float(yaw_degrees)


def is_head_down(kps, pitch_degrees=30):
    _, pitch = estimate_yaw_pitch(kps)
    if pitch is None:
        return False
    return pitch >= float(pitch_degrees)


def is_gaze_away(kps, yaw_degrees=25, pitch_degrees=25):
    """不看屏幕：偏头或低头超过阈值。"""
    yaw, pitch = estimate_yaw_pitch(kps)
    if yaw is None:
        return False
    return abs(yaw) >= float(yaw_degrees) or pitch >= float(pitch_degrees)


def is_chin_rest(kps):
    """手腕靠近下颌（鼻/嘴附近）。"""
    if not _kp_ok(kps, NOSE):
        return False
    nose = _kp(kps, NOSE)
    sw = _shoulder_width(kps)
    for wi in (L_WRIST, R_WRIST):
        if not _kp_ok(kps, wi):
            continue
        wx, wy = _kp(kps, wi)
        dist = math.hypot(wx - nose[0], wy - nose[1]) / sw
        if dist < 0.55 and wy >= nose[1] - 0.2 * sw:
            return True
    return False


def is_cover_mouth(kps):
    """捂嘴：手腕贴近口鼻前方（略高于下巴托）。"""
    if not _kp_ok(kps, NOSE):
        return False
    nose = _kp(kps, NOSE)
    sw = _shoulder_width(kps)
    for wi in (L_WRIST, R_WRIST):
        if not _kp_ok(kps, wi):
            continue
        wx, wy = _kp(kps, wi)
        dist = math.hypot(wx - nose[0], wy - nose[1]) / sw
        # 更贴近口部：水平接近、竖直略低于鼻
        if dist < 0.42 and abs(wy - nose[1]) < 0.35 * sw and wy >= nose[1] - 0.15 * sw:
            return True
    return False


def is_wrist_near_ear(kps, dist_ratio=0.45):
    """手腕靠近任一耳（镜脚/太阳穴区域）。"""
    sw = _shoulder_width(kps)
    ears = []
    for ei in (L_EAR, R_EAR):
        if _kp_ok(kps, ei):
            ears.append(_kp(kps, ei))
    if not ears:
        return False
    for wi in (L_WRIST, R_WRIST):
        if not _kp_ok(kps, wi):
            continue
        wx, wy = _kp(kps, wi)
        for ex, ey in ears:
            if math.hypot(wx - ex, wy - ey) / sw < float(dist_ratio):
                return True
    return False


def is_standing(kps, box=None):
    """站立：踝明显低于髋，或框高宽比较大。"""
    if _kp_ok(kps, L_HIP) and _kp_ok(kps, L_ANKLE):
        if _kp(kps, L_ANKLE)[1] > _kp(kps, L_HIP)[1] + 25:
            return True
    if _kp_ok(kps, R_HIP) and _kp_ok(kps, R_ANKLE):
        if _kp(kps, R_ANKLE)[1] > _kp(kps, R_HIP)[1] + 25:
            return True
    if box is not None:
        w = max(float(box.x2 - box.x1), 1.0)
        h = max(float(box.y2 - box.y1), 1.0)
        if h / w >= 1.45:
            return True
    return False


def is_fall(kps, box=None):
    """倒地：肩-髋连线接近水平，或检测框宽高比偏大且重心偏低。"""
    if _kp_ok(kps, L_SHOULDER) and _kp_ok(kps, R_SHOULDER) and _kp_ok(kps, L_HIP) and _kp_ok(kps, R_HIP):
        sy = (_kp(kps, L_SHOULDER)[1] + _kp(kps, R_SHOULDER)[1]) / 2.0
        hy = (_kp(kps, L_HIP)[1] + _kp(kps, R_HIP)[1]) / 2.0
        sx = (_kp(kps, L_SHOULDER)[0] + _kp(kps, R_SHOULDER)[0]) / 2.0
        hx = (_kp(kps, L_HIP)[0] + _kp(kps, R_HIP)[0]) / 2.0
        dx, dy = abs(hx - sx), abs(hy - sy)
        if dx > 1 and dy / max(dx, 1.0) < 0.45:
            return True
    if box is not None:
        w = max(float(box.x2 - box.x1), 1.0)
        h = max(float(box.y2 - box.y1), 1.0)
        if w / h > 1.35:
            return True
    return False


def is_finger_point(kps, require_standing=True):
    """站立 + 手腕明显高于肘（伸臂指向）。"""
    standing = True
    if require_standing and _kp_ok(kps, L_HIP) and _kp_ok(kps, L_ANKLE):
        standing = _kp(kps, L_ANKLE)[1] > _kp(kps, L_HIP)[1] + 20
    if require_standing and not standing:
        return False
    for elbow, wrist in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
        if _kp_ok(kps, elbow) and _kp_ok(kps, wrist):
            if _kp(kps, wrist)[1] < _kp(kps, elbow)[1] - 15:
                return True
    return False


# 逐人布尔判定（不含需跨帧计数的特殊类型）
BEHAVIOR_EVALUATORS = {
    'gaze_away': lambda kps, box, spec: is_gaze_away(
        kps, spec.get('yaw_degrees', 25), spec.get('pitch_degrees', 25)
    ),
    'look_aside': lambda kps, box, spec: is_looking_aside(kps, spec.get('yaw_degrees', 45)),
    'head_down': lambda kps, box, spec: is_head_down(kps, spec.get('pitch_degrees', 30)),
    'chin_rest': lambda kps, box, spec: is_chin_rest(kps),
    'cover_mouth': lambda kps, box, spec: is_cover_mouth(kps),
    'fall': lambda kps, box, spec: is_fall(kps, box),
    'finger_point': lambda kps, box, spec: is_finger_point(
        kps, bool(spec.get('require_standing', True))
    ),
}

# 需要在主循环里做跨帧状态的类型
STATEFUL_BEHAVIORS = frozenset({'smart_glasses', 'invigilator_absent'})


class PoseBehaviorAlgorithm(BaseAlgorithm):
    """一次姿态推理，多条行为规则计时告警。"""

    def _extract_persons(self, result, logger):
        """从 DetectionResult 提取 (box, keypoints[17][3]) 列表。"""
        persons = []
        boxes = getattr(result, 'boxes', None) or []
        keypoints = getattr(result, 'keypoints', None)
        if keypoints is None and hasattr(result, '_raw') and result._raw is not None:
            try:
                raw = result._raw[0] if isinstance(result._raw, list) else result._raw
                if hasattr(raw, 'keypoints') and raw.keypoints is not None:
                    kpdata = raw.keypoints.data
                    if hasattr(kpdata, 'cpu'):
                        kpdata = kpdata.cpu().numpy()
                    for i, box in enumerate(boxes):
                        kps = kpdata[i] if i < len(kpdata) else None
                        persons.append((box, kps))
                    return persons
            except Exception as e:
                logger.debug(f"keypoints from raw failed: {e}")

        if keypoints is not None:
            for i, box in enumerate(boxes):
                kps = keypoints[i] if i < len(keypoints) else None
                persons.append((box, kps))
            return persons

        for box in boxes:
            persons.append((box, None))
        return persons

    def _eval_smart_glasses(self, persons, spec, state, now):
        """
        手腕靠近耳朵计为一次接触；离开后再靠近累加次数。
        在 seconds 窗口内达到 min_touches 则触发。
        """
        min_touches = int(spec.get('min_touches', 3))
        window = float(spec.get('seconds', 30))
        near = False
        hit_box = None
        for box, kps in persons:
            if kps is None:
                continue
            if is_wrist_near_ear(kps, spec.get('ear_dist_ratio', 0.45)):
                near = True
                hit_box = box
                break

        if near and not state.get('near'):
            state['touches'] = int(state.get('touches') or 0) + 1
            state['window_start'] = state.get('window_start') or now
        if not near:
            state['near'] = False
        else:
            state['near'] = True

        ws = state.get('window_start')
        if ws and (now - ws).total_seconds() > window:
            state['touches'] = 1 if near else 0
            state['window_start'] = now if near else None

        if int(state.get('touches') or 0) >= min_touches:
            state['touches'] = 0
            state['window_start'] = None
            state['near'] = False
            return True, hit_box
        return False, None

    def _eval_invigilator_absent(self, persons, spec, state, now):
        """站立人数持续少于 min_standing 达到 seconds（默认 120s）则告警。"""
        min_standing = int(spec.get('min_standing', 2))
        seconds = float(spec.get('seconds', 120))
        standing_n = 0
        sample_box = None
        for box, kps in persons:
            if kps is None:
                # 无关键点时用框粗判
                if box is not None and is_standing(None, box):
                    standing_n += 1
                    sample_box = sample_box or box
                continue
            if is_standing(kps, box):
                standing_n += 1
                sample_box = sample_box or box

        if standing_n >= min_standing:
            state['since'] = None
            return False, None, standing_n

        if state.get('since') is None:
            state['since'] = now
            return False, None, standing_n

        elapsed = (now - state['since']).total_seconds()
        if elapsed < seconds:
            return False, None, standing_n
        state['since'] = now
        return True, sample_box, standing_n

    def process(self, camera_stream, config_dict, logger, stop_event, on_alert, runtime):
        try:
            parameters = config_dict.get('parameters', {})
            model_path = config_dict.get('model_local_path')
            algo_params = parameters.get('algorithm_parameters') or {}
            confidence = float(parameters.get('confidence', 0.5))
            alert_threshold = int(parameters.get('alertThreshold', 10))
            behaviors = [b for b in (algo_params.get('behaviors') or []) if b.get('enabled', True)]
            task_name = config_dict.get('task_id', 'unknown_task')
            task_sched_start = parameters.get('schedule_start') or algo_params.get('schedule_start')
            task_sched_end = parameters.get('schedule_end') or algo_params.get('schedule_end')

            if not behaviors:
                logger.warning("pose_behavior: no behaviors configured")
                return
            if camera_stream is None:
                logger.error("error: camera is None")
                return

            rtsp_url = config_dict.get('camera', {}).get('rtsp_url')
            infer_fps = float(
                parameters.get('inferFps')
                or algo_params.get('infer_fps')
                or 5
            )
            infer_interval = (1.0 / infer_fps) if infer_fps > 0 else 0.0
            decode_fps = infer_fps if infer_fps > 0 else 12.0
            pump = FramePump(camera_stream, stop_event, rtsp_url, logger, max_decode_fps=decode_fps)
            pump.start()
            try:
                runtime.load(model_path)
                logger.info(
                    f"Throughput caps: inferFps={infer_fps:.1f} decodeFps≈{decode_fps:.1f}"
                )
                first_frame, last_seq = pump.get_latest(last_seq=0, wait_sec=8.0)
                if first_frame is None:
                    import cv2
                    h, w = camera_stream.get(cv2.CAP_PROP_FRAME_HEIGHT), camera_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
                else:
                    h, w = first_frame.shape[0], first_frame.shape[1]
                new_h, new_w, top, bottom, left, right = get_letterbox_params(h, w, target_size=640)
                if new_h is None:
                    logger.error("error: get_letterbox_params return None")
                    return

                state_since = {b.get('id') or b.get('type'): None for b in behaviors}
                last_alert = {b.get('id') or b.get('type'): None for b in behaviors}
                extra_state = {b.get('id') or b.get('type'): {} for b in behaviors}
                next_infer_t = 0.0
                frame_idx = 0
                last_log = time.time()
                fps_window_t = time.time()
                fps_window_n = 0

                while not stop_event.is_set():
                    now_t = time.time()
                    if infer_interval and now_t < next_infer_t:
                        if stop_event.wait(timeout=min(0.05, next_infer_t - now_t)):
                            break
                        continue

                    frame, last_seq = pump.get_latest(last_seq=last_seq, wait_sec=2.0)
                    if stop_event.is_set():
                        break
                    if frame is None or not is_valid_frame(frame):
                        continue
                    next_infer_t = time.time() + infer_interval
                    processed = preprocess(frame, new_h, new_w, top, bottom, left, right)
                    if processed is None:
                        continue
                    t0 = time.time()
                    try:
                        result = runtime.infer(processed, conf=confidence, classes=None, imgsz=640)
                    except Exception as e:
                        logger.warning(f"pose infer skipped: {e}")
                        continue
                    infer_ms = (time.time() - t0) * 1000
                    frame_idx += 1
                    fps_window_n += 1
                    if frame_idx == 1 or time.time() - last_log >= 5:
                        dt = max(time.time() - fps_window_t, 1e-6)
                        logger.info(
                            f"Frame #{frame_idx} process={fps_window_n / dt:.1f}fps "
                            f"infer={infer_ms:.0f}ms behaviors={len(behaviors)}"
                        )
                        last_log = time.time()
                        fps_window_t = last_log
                        fps_window_n = 0

                    persons = self._extract_persons(result, logger)
                    now = datetime.now()

                    for spec in behaviors:
                        bid = spec.get('id') or spec.get('type')
                        btype = spec.get('type')
                        if not is_item_active_now(now, spec, task_sched_start, task_sched_end):
                            state_since[bid] = None
                            extra_state[bid] = {}
                            continue

                        alert_type = spec.get('alert_type') or btype
                        hit_person = None
                        message = None

                        if btype == 'smart_glasses':
                            ok, box = self._eval_smart_glasses(
                                persons, spec, extra_state[bid], now
                            )
                            if ok:
                                hit_person = (box, None)
                                message = (
                                    f"行为[{spec.get('name') or btype}] "
                                    f"镜脚触碰达 {spec.get('min_touches', 3)} 次"
                                )
                        elif btype == 'invigilator_absent':
                            ok, box, standing_n = self._eval_invigilator_absent(
                                persons, spec, extra_state[bid], now
                            )
                            if ok:
                                hit_person = (box, None)
                                message = (
                                    f"行为[{spec.get('name') or btype}] "
                                    f"站立仅 {standing_n} 人（需≥{spec.get('min_standing', 2)}）"
                                )
                        else:
                            evaluator = BEHAVIOR_EVALUATORS.get(btype)
                            if not evaluator:
                                continue
                            seconds = float(spec.get('seconds', 3))
                            for box, kps in persons:
                                if kps is None and btype not in ('fall',):
                                    continue
                                try:
                                    if evaluator(kps, box, spec):
                                        hit_person = (box, kps)
                                        break
                                except Exception as e:
                                    logger.debug(f"behavior {btype} eval error: {e}")

                            if hit_person is None:
                                state_since[bid] = None
                                continue
                            if state_since[bid] is None:
                                state_since[bid] = now
                                continue
                            elapsed = (now - state_since[bid]).total_seconds()
                            if elapsed < seconds:
                                continue
                            message = f"行为[{spec.get('name') or btype}] 持续 {elapsed:.1f}s"

                        if hit_person is None:
                            continue
                        la = last_alert[bid]
                        if la and (now - la).total_seconds() < alert_threshold:
                            continue

                        box, _ = hit_person
                        alert_frame = self.draw_alert_overlay(
                            processed,
                            boxes=[box] if box is not None else None,
                            caption=message or f"行为[{spec.get('name') or btype}]",
                        )
                        logger.info(f"Triggering {alert_type} for {task_name}")
                        if on_alert:
                            on_alert(
                                alert_type=alert_type,
                                confidence=float(getattr(box, 'confidence', 1.0) or 1.0) if box else 1.0,
                                image_frame=alert_frame,
                                message=message,
                            )
                        last_alert[bid] = now
                        if btype not in STATEFUL_BEHAVIORS:
                            state_since[bid] = now
            finally:
                pump.release()
        except Exception as e:
            logger.error(f"Error in pose_behavior: {str(e)}", exc_info=True)
            raise
