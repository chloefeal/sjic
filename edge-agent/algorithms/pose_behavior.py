"""
姿态行为引擎：张望 / 低头 / 不看屏幕 / 手指屏幕 /
捂嘴报题 / AI 智能眼镜 / 手指点击手表 / 考试员不巡场 /
人员缺席 / 人员滞留 / 通道聚集。
智能眼镜和智能手表是融合场景，混合任务里才和目标检测一起判定。
点状违规和眼镜、手表按人计时：同一次推理里每人一个轨迹 id。
缺席、滞留、聚集、考试员不巡场仍按区域计数，不按人跟踪。

依赖姿态模型（YOLO-Pose 等）经 runtime.infer 返回 keypoints。
几何判断按摄像头 mount_position（前/后/侧/正上方）切换，避免把俯视看屏判成违规。
若当前 runtime 仅返回 boxes，则跳过关键点行为；人数统计仍可用这些框。
"""
import math
import time
from datetime import datetime

from .base import BaseAlgorithm, effective_confidence
from .detection_preview import offer_detection_preview
from .object_detection import ObjectDetectionAlgorithm
from .pose_geometry import (
    L_EAR,
    L_EYE,
    L_HIP,
    L_SHOULDER,
    NOSE,
    R_EAR,
    R_EYE,
    R_HIP,
    R_SHOULDER,
    _conf,
    _head_axis_ratios,
    _head_hip_ratio,
    _kp,
    _midpoint,
    _neck_ratio,
    accumulate_rising_count,
    ear_asymmetry,
    estimate_yaw_pitch,
    face_visible,
    is_cover_mouth,
    is_finger_click_watch,
    is_finger_point,
    is_gaze_away,
    is_head_down,
    is_looking_aside,
    is_standing,
    is_wrist_near_ear,
    normalize_mount_position,
)
from .rules import (
    PERSON_COUNT_TYPES,
    behavior_rois,
    build_rules,
    filter_persons_roi,
    person_count_message,
    person_count_specs,
)
from utils.calc import get_letterbox_params, preprocess
from utils.rtsp import FramePump, is_valid_frame
from utils.schedule import is_item_active_now


def _debug_pose_decision(btype, kps, box, spec, view, matched, elapsed, track_id, seconds, scene_conf):
    # #region agent log
    try:
        import json
        from pathlib import Path
        yaw_ref = float(spec.get('yaw_degrees', 45 if btype == 'look_aside' else 25) or 0)
        pitch_ref = float(spec.get('pitch_degrees', 25) or 0)
        scale = max(yaw_ref, 1.0) / 45.0
        along, perp = _head_axis_ratios(kps)
        yaw, pitch = estimate_yaw_pitch(kps)
        neck = _neck_ratio(kps)
        head_hip = _head_hip_ratio(kps)

        def r(v):
            return None if v is None else round(float(v), 3)

        box_xy = None
        conf = None
        if box is not None:
            box_xy = [r(getattr(box, 'x1', None)), r(getattr(box, 'y1', None)),
                      r(getattr(box, 'x2', None)), r(getattr(box, 'y2', None))]
            conf = r(getattr(box, 'confidence', None))
        kp = {}
        for name, idx in (('nose', NOSE), ('leye', L_EYE), ('reye', R_EYE), ('lear', L_EAR), ('rear', R_EAR),
                          ('lsh', L_SHOULDER), ('rsh', R_SHOULDER)):
            if kps is not None and idx < len(kps):
                kp[name] = [r(kps[idx][0]), r(kps[idx][1]), r(kps[idx][2])]
        row = {
            'sessionId': '1e610f', 'runId': 'pre-fix', 'hypothesisId': 'ABCDE',
            'location': 'pose_behavior.py:pointwise', 'message': 'pose decision',
            'timestamp': int(time.time() * 1000),
            'data': {
                'btype': btype, 'view': view, 'track': track_id, 'matched': bool(matched),
                'elapsed': r(elapsed), 'need_s': r(seconds), 'scene_conf': r(scene_conf),
                'box': box_xy, 'conf': conf, 'yaw_ref': r(yaw_ref), 'pitch_ref': r(pitch_ref),
                'face': bool(face_visible(kps)), 'along': r(along), 'perp': r(perp),
                'shift_gate': r(0.28 * scale), 'strong_gate': r(0.42 * scale),
                'ear_gate': r(0.40 * scale),
                'ear_sum': r(_conf(kps, L_EAR) + _conf(kps, R_EAR)),
                'ear_asym': r(ear_asymmetry(kps)), 'neck': r(neck), 'head_hip': r(head_hip),
                'yaw': r(yaw), 'pitch': r(pitch),
                'aside': bool(is_looking_aside(kps, yaw_ref, view)),
                'head_down': bool(is_head_down(kps, pitch_ref, view)),
                'kp': kp,
            },
        }
        path = Path(__file__).resolve().parents[2] / 'debug-1e610f.log'
        with path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    except Exception:
        pass
    # #endregion


def _eval_behavior(btype, kps, box, spec, view):
    if btype == 'gaze_away':
        return is_gaze_away(
            kps, spec.get('yaw_degrees', 25), spec.get('pitch_degrees', 25), view
        )
    if btype == 'look_aside':
        return is_looking_aside(kps, spec.get('yaw_degrees', 45), view)
    if btype == 'head_down':
        return is_head_down(kps, spec.get('pitch_degrees', 30), view)
    if btype == 'cover_mouth':
        return is_cover_mouth(kps, view)
    if btype == 'finger_point':
        return is_finger_point(kps, bool(spec.get('require_standing', True)), view)
    return False


POINTWISE_TYPES = frozenset({
    'gaze_away', 'look_aside', 'head_down',
    'cover_mouth', 'finger_point',
})
_DROPPED_BEHAVIOR_TYPES = frozenset({'chin_rest'})
STATEFUL_BEHAVIORS = frozenset({'smart_glasses', 'smart_watch', 'invigilator_absent'})

# 连续这么多次推理都没再检出这个人，轨迹仍保留。大约 1 次/秒时就是两秒。
TRACK_GRACE_MISSES = 2
_SHOULDER_GATE = 0.75
_BOX_GATE = 0.5


def _pose_points(kps):
    """只有 COCO 17 点才用来取肩、髋。短列表和字符串都当没有关键点。"""
    if kps is None or isinstance(kps, (str, bytes)):
        return None
    try:
        count = len(kps)
    except TypeError:
        return None
    if count < 17:
        return None
    return kps


def _box_center_width(box):
    try:
        x1, y1 = float(box.x1), float(box.y1)
        x2, y2 = float(box.x2), float(box.y2)
    except (TypeError, ValueError, AttributeError):
        return None, None
    if x2 <= x1 or y2 <= y1:
        return None, None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0), (x2 - x1)


def _person_anchor(box, kps):
    """锚点和尺度。双肩中点优先，否则髋中点，否则框中心。有双肩时尺度是肩宽，否则用框宽。"""
    points = _pose_points(kps)
    if points is not None:
        shoulders = _midpoint(points, L_SHOULDER, R_SHOULDER)
        if shoulders is not None:
            left = _kp(points, L_SHOULDER)
            right = _kp(points, R_SHOULDER)
            width = max(math.hypot(left[0] - right[0], left[1] - right[1]), 1.0)
            return shoulders, width, True
        hips = _midpoint(points, L_HIP, R_HIP)
        if hips is not None:
            _center, width = _box_center_width(box)
            if width is None:
                left = _kp(points, L_HIP)
                right = _kp(points, R_HIP)
                width = max(math.hypot(left[0] - right[0], left[1] - right[1]), 1.0)
            return hips, width, False
    center, width = _box_center_width(box)
    if center is not None:
        return center, width, False
    return (0.0, 0.0), None, False


def _person_key(box, kps):
    if box is not None:
        return ('box', id(box))
    return ('kps', id(kps))


def _match_gate(track_scale, track_shoulder, det_scale, det_shoulder):
    if track_shoulder or det_shoulder:
        scales = []
        if track_shoulder and track_scale:
            scales.append(track_scale)
        if det_shoulder and det_scale:
            scales.append(det_scale)
        if scales:
            return _SHOULDER_GATE * max(scales)
    scales = [scale for scale in (track_scale, det_scale) if scale]
    if not scales:
        return 1.0
    return _BOX_GATE * max(scales)


class _Track:
    def __init__(self, track_id, anchor, scale, use_shoulder, seen):
        self.id = track_id
        self.anchor = anchor
        self.scale = scale
        self.use_shoulder = use_shoulder
        self.seen = seen
        self.misses = 0
        self.since = {}
        self.last_alert = {}
        self.touch = {}


class PersonTracker:
    """用肩中点把相邻推理结果里的同一个人接起来。不加载跟踪模型。"""

    def __init__(self):
        self.tracks = []
        self._next_id = 1
        self._by_key = {}

    def track_for(self, box, kps):
        return self._by_key.get(_person_key(box, kps))

    def clear_behavior(self, behavior_id):
        for track in self.tracks:
            track.since.pop(behavior_id, None)
            track.touch.pop(behavior_id, None)

    def update(self, persons, now):
        detections = []
        for box, kps in persons or []:
            anchor, scale, use_shoulder = _person_anchor(box, kps)
            detections.append((box, kps, anchor, scale, use_shoulder))

        # 上一帧还在的人直接按位置接上，不看两次推理隔了多久。
        # 中间没检出时按帧数留宽限，推理大约 1 次/秒时一帧漏检也不会清掉计时。
        pairs = []
        for det_index, det in enumerate(detections):
            anchor = det[2]
            for track_index, track in enumerate(self.tracks):
                dist = math.hypot(anchor[0] - track.anchor[0], anchor[1] - track.anchor[1])
                gate = _match_gate(track.scale, track.use_shoulder, det[3], det[4])
                if dist < gate:
                    pairs.append((dist, det_index, track_index))
        pairs.sort()
        used_dets = set()
        used_tracks = set()
        assigned = {}
        for _dist, det_index, track_index in pairs:
            if det_index in used_dets or track_index in used_tracks:
                continue
            used_dets.add(det_index)
            used_tracks.add(track_index)
            assigned[det_index] = self.tracks[track_index]

        current = []
        for det_index, det in enumerate(detections):
            _box, _kps, anchor, scale, use_shoulder = det
            track = assigned.get(det_index)
            if track is None:
                track = _Track(self._next_id, anchor, scale, use_shoulder, now)
                self._next_id += 1
            else:
                track.anchor = anchor
                track.scale = scale
                track.use_shoulder = use_shoulder
                track.seen = now
                track.misses = 0
            current.append(track)

        matched_ids = {track.id for track in current}
        unmatched = []
        for track in self.tracks:
            if track.id in matched_ids:
                continue
            track.misses += 1
            if track.misses <= TRACK_GRACE_MISSES:
                unmatched.append(track)
        self.tracks = unmatched + current
        self._by_key = {
            _person_key(box, kps): track
            for (box, kps, _anchor, _scale, _shoulder), track in zip(detections, current)
        }
        return current


class PoseBehaviorAlgorithm(BaseAlgorithm):
    """一次姿态推理，多条行为规则计时告警。"""

    def __init__(self):
        self._person_tracker = PersonTracker()

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

    def _rising_hits(
        self, people, bid, now, active_fn, min_touches, window_seconds,
        hold_contact=False, logger=None,
    ):
        """按轨迹累计从无到有的次数。每个人各自计数，达到次数的都返回。"""
        hits = []
        for box, kps in people or []:
            track = self._person_tracker.track_for(box, kps)
            if track is None:
                continue
            state = track.touch.setdefault(bid, {})
            try:
                active = bool(active_fn(box, kps))
            except Exception as e:
                if logger is not None:
                    logger.debug(f"touch eval error: {e}")
                active = False
            if accumulate_rising_count(
                state, active, now, min_touches, window_seconds, hold_contact=hold_contact,
            ):
                hits.append((box, kps, track))
        return hits

    def _glasses_hits(self, people, spec, bid, now, logger=None):
        ratio = spec.get('ear_dist_ratio', 0.28)
        return self._rising_hits(
            people, bid, now,
            lambda _box, kps: kps is not None and is_wrist_near_ear(kps, ratio),
            spec.get('min_touches', 3),
            spec.get('seconds', 30),
            logger=logger,
        )

    def _watch_hits(self, people, watch_boxes, spec, bid, now, logger=None):
        ratio = spec.get('watch_dist_ratio', 0.50)
        return self._rising_hits(
            people, bid, now,
            lambda _box, kps: is_finger_click_watch(kps, watch_boxes, ratio),
            spec.get('min_touches', 1),
            spec.get('seconds', 30),
            hold_contact=True,
            logger=logger,
        )

    def _eval_invigilator_absent(self, persons, spec, state, now, view):
        """站立人数持续少于 min_standing 达到 seconds（默认 120s）则告警。"""
        min_standing = int(spec.get('min_standing', 2))
        seconds = float(spec.get('seconds', 120))
        standing_n = 0
        sample_box = None
        for box, kps in persons:
            if kps is None:
                if box is not None and is_standing(None, box, view):
                    standing_n += 1
                    sample_box = sample_box or box
                continue
            if is_standing(kps, box, view):
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

    def evaluate_behaviors(
        self, behaviors, persons, now, processed, view, task_confidence,
        task_sched_start, task_sched_end, alert_threshold, task_name,
        state_since, last_alert, extra_state, person_rules, scene_rois,
        on_alert, logger, algorithm_id=None, pose_only=False,
    ):
        """
        姿态场景判定，供姿态任务和混合任务共用。

        先把本帧每个人接到自己的轨迹上。点状场景按这条轨迹计时，
        每人达标各报一次，冷却也只挡住这个人。人数场景仍交给区域计数。
        pose_only 时才处理单独姿态任务里的眼镜，并忽略跌倒和手表。
        """
        self._person_tracker.update(persons, now)
        for spec in behaviors:
            bid = spec.get('id') or spec.get('type')
            btype = spec.get('type')
            if not is_item_active_now(now, spec, task_sched_start, task_sched_end):
                state_since[bid] = None
                extra_state[bid] = {}
                self._person_tracker.clear_behavior(bid)
                paused = person_rules.get(bid)
                if paused is not None:
                    paused._state_since = None
                continue

            if pose_only and btype == 'fall':
                if not extra_state[bid].get('warned'):
                    logger.warning("人员倒地已改为目标检测场景，姿态任务中的跌倒行为已忽略")
                    extra_state[bid]['warned'] = True
                continue
            if pose_only and btype == 'smart_watch':
                if not extra_state[bid].get('warned'):
                    logger.warning("智能手表作弊检测是融合场景，请在混合任务里配置手表类别")
                    extra_state[bid]['warned'] = True
                continue

            scene_conf = effective_confidence(spec, task_confidence)
            roi_polygon, roi_points = scene_rois.get(bid, (None, None))
            persons_for_spec = filter_persons_roi(
                [
                    (box, kps) for box, kps in persons
                    if box is None or float(getattr(box, 'confidence', 0) or 0) >= scene_conf
                ],
                roi_polygon,
            )
            hit_person = None
            message = None

            if pose_only and btype == 'smart_glasses':
                hits = self._glasses_hits(persons_for_spec, spec, bid, now, logger)
                message = (
                    f"行为[{spec.get('name') or btype}] "
                    f"镜脚触碰达 {spec.get('min_touches', 3)} 次"
                )
                for box, kps, track in hits:
                    self.emit_pose_alert(
                        spec, btype, bid, (box, kps), message, persons_for_spec, processed,
                        now, alert_threshold, task_name, algorithm_id, track.last_alert, track.since,
                        on_alert, logger, roi_points=roi_points,
                    )
                continue
            elif btype == 'invigilator_absent':
                ok, box, standing_n = self._eval_invigilator_absent(
                    persons_for_spec, spec, extra_state[bid], now, view
                )
                if ok:
                    hit_person = (box, None)
                    message = (
                        f"行为[{spec.get('name') or btype}] "
                        f"站立仅 {standing_n} 人（需≥{spec.get('min_standing', 2)}）"
                    )
            elif btype in PERSON_COUNT_TYPES:
                rule = person_rules.get(bid)
                if rule is None:
                    continue
                person_boxes = [box for box, _kps in persons_for_spec if box is not None]
                hit = rule.evaluate(person_boxes, now, logger)
                if not hit:
                    continue
                message = person_count_message(spec, btype, hit)
                alert_frame = self.draw_alert_overlay(
                    processed,
                    boxes=hit['boxes'] or None,
                    roi_points=rule.roi_points,
                    caption=message,
                )
                logger.info(f"Triggering {spec.get('alert_type') or btype} for {task_name}")
                if on_alert:
                    on_alert(
                        alert_type=spec.get('alert_type') or btype,
                        confidence=hit['confidence'],
                        image_frame=alert_frame,
                        message=message,
                        raw_frame=processed,
                        algorithm_id=algorithm_id,
                    )
                last_alert[bid] = now
                continue
            elif btype in POINTWISE_TYPES:
                seconds = float(spec.get('seconds', 3))
                in_scene = set()
                ready = []
                # #region agent log
                if btype in ('gaze_away', 'look_aside'):
                    _kept = {id(box) for box, _kps in persons_for_spec if box is not None}
                    _dropped = []
                    for _box, _kps in persons or []:
                        if _box is None or id(_box) in _kept:
                            continue
                        _dropped.append([
                            round(float(_box.x1), 1), round(float(_box.y1), 1),
                            round(float(_box.x2), 1), round(float(_box.y2), 1),
                            round(float(getattr(_box, 'confidence', 0) or 0), 3),
                        ])
                    if _dropped:
                        try:
                            import json
                            from pathlib import Path
                            _row = {
                                'sessionId': '1e610f', 'runId': 'pre-fix', 'hypothesisId': 'G',
                                'location': 'pose_behavior.py:filter', 'message': 'dropped persons',
                                'timestamp': int(time.time() * 1000),
                                'data': {
                                    'btype': btype, 'view': view, 'scene_conf': scene_conf,
                                    'dropped': _dropped,
                                },
                            }
                            _path = Path(__file__).resolve().parents[2] / 'debug-1e610f.log'
                            with _path.open('a', encoding='utf-8') as _fh:
                                _fh.write(json.dumps(_row, ensure_ascii=False) + '\n')
                        except Exception:
                            pass
                # #endregion
                for box, kps in persons_for_spec:
                    track = self._person_tracker.track_for(box, kps)
                    if track is None:
                        continue
                    in_scene.add(track.id)
                    if kps is None:
                        track.since.pop(bid, None)
                        continue
                    try:
                        matched = _eval_behavior(btype, kps, box, spec, view)
                    except Exception as e:
                        logger.debug(f"behavior {btype} eval error: {e}")
                        matched = False
                    # #region agent log
                    if btype in ('gaze_away', 'look_aside'):
                        _started = track.since.get(bid) if matched else None
                        _elapsed = (now - _started).total_seconds() if _started is not None else None
                        _debug_pose_decision(
                            btype, kps, box, spec, view, matched, _elapsed, track.id, seconds, scene_conf,
                        )
                    # #endregion
                    if not matched:
                        track.since.pop(bid, None)
                        continue
                    started = track.since.get(bid)
                    if started is None:
                        track.since[bid] = now
                        continue
                    elapsed = (now - started).total_seconds()
                    if elapsed < seconds:
                        continue
                    ready.append((track, box, kps, elapsed))
                for track in self._person_tracker.tracks:
                    if track.seen == now and track.id not in in_scene:
                        track.since.pop(bid, None)
                for track, box, kps, elapsed in ready:
                    self.emit_pose_alert(
                        spec, btype, bid, (box, kps),
                        f"行为[{spec.get('name') or btype}] 持续 {elapsed:.1f}s",
                        persons_for_spec, processed,
                        now, alert_threshold, task_name, algorithm_id,
                        track.last_alert, track.since,
                        on_alert, logger, roi_points=roi_points,
                    )
                continue
            else:
                continue

            self.emit_pose_alert(
                spec, btype, bid, hit_person, message, persons_for_spec, processed,
                now, alert_threshold, task_name, algorithm_id, last_alert, state_since,
                on_alert, logger, roi_points=roi_points,
            )

    def emit_pose_alert(
        self, spec, btype, bid, hit_person, message, persons_for_spec, processed,
        now, alert_threshold, task_name, algorithm_id, last_alert, state_since,
        on_alert, logger, extra_boxes=None, roi_points=None,
    ):
        """命中后按传入的冷却画框并告警。点状行为告警后重新计时，有状态场景不重置。"""
        if hit_person is None:
            return
        la = last_alert.get(bid)
        if la and (now - la).total_seconds() < alert_threshold:
            return
        box, kps = hit_person
        if btype == 'invigilator_absent':
            skeletons = [pk for _, pk in persons_for_spec if pk is not None]
        elif kps is not None:
            skeletons = [kps]
        else:
            skeletons = None
        draw_boxes = []
        if box is not None:
            draw_boxes.append(box)
        for extra in extra_boxes or []:
            if extra is not box:
                draw_boxes.append(extra)
        alert_frame = self.draw_alert_overlay(
            processed,
            boxes=draw_boxes or None,
            roi_points=roi_points,
            caption=message or f"行为[{spec.get('name') or btype}]",
            skeletons=skeletons,
        )
        logger.info(f"Triggering {spec.get('alert_type') or btype} for {task_name}")
        if on_alert:
            on_alert(
                alert_type=spec.get('alert_type') or btype,
                confidence=float(getattr(box, 'confidence', 1.0) or 1.0) if box else 1.0,
                image_frame=alert_frame,
                message=message,
                raw_frame=processed,
                algorithm_id=algorithm_id,
            )
        last_alert[bid] = now
        if btype not in STATEFUL_BEHAVIORS and bid in state_since:
            state_since[bid] = now

    def process(self, camera_stream, config_dict, logger, stop_event, on_alert, runtime):
        try:
            parameters = config_dict.get('parameters', {})
            model_path = config_dict.get('model_local_path')
            algo_params = parameters.get('algorithm_parameters') or {}
            task_confidence = float(parameters.get('confidence', 0.5))
            alert_threshold = int(parameters.get('alertThreshold', 10))
            behaviors = [
                b for b in (algo_params.get('behaviors') or [])
                if b.get('enabled', True) and b.get('type') not in _DROPPED_BEHAVIOR_TYPES
            ]
            confidence = min(
                (effective_confidence(b, task_confidence) for b in behaviors),
                default=task_confidence,
            )
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
            view = normalize_mount_position(
                (config_dict.get('camera') or {}).get('mount_position')
            )
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
                    f"Throughput caps: inferFps={infer_fps:.1f} decodeFps≈{decode_fps:.1f} "
                    f"mount_position={view}"
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
                od_helper = ObjectDetectionAlgorithm()

                def transform_roi(region):
                    return od_helper._transform_roi(region, new_h, new_w, top, left, logger)

                count_rules = {
                    rule.id: rule
                    for rule in build_rules(
                        person_count_specs(behaviors),
                        transform_roi,
                        alert_threshold,
                        None,
                        logger,
                    )
                }
                scene_rois = behavior_rois(behaviors, transform_roi)
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
                    offer_detection_preview(
                        config_dict.get('task_id'),
                        processed,
                        persons=persons,
                        pose_threshold=confidence,
                        infer_ms=infer_ms,
                    )
                    now = datetime.now()
                    self.evaluate_behaviors(
                        behaviors, persons, now, processed, view, task_confidence,
                        task_sched_start, task_sched_end, alert_threshold, task_name,
                        state_since, last_alert, extra_state, count_rules, scene_rois,
                        on_alert, logger, pose_only=True,
                    )
            finally:
                pump.release()
        except Exception as e:
            logger.error(f"Error in pose_behavior: {str(e)}", exc_info=True)
            raise
