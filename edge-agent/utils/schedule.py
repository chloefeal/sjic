"""每日运行时段：HH:MM，支持跨午夜；场景优先于任务。"""
from datetime import datetime, time as dtime


def parse_hhmm(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(':')
    if len(parts) < 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except (TypeError, ValueError):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return dtime(hour, minute)


def in_daily_window(now, start_hhmm, end_hhmm):
    """
    当前时刻是否落在 [start, end) 每日窗口内。
    未配置完整时段 → 视为始终有效。
    跨午夜：如 22:00–06:00。
    """
    start = parse_hhmm(start_hhmm)
    end = parse_hhmm(end_hhmm)
    if start is None or end is None:
        return True
    if isinstance(now, datetime):
        cur = now.time().replace(second=0, microsecond=0)
    else:
        cur = now
    if start == end:
        return True
    if start < end:
        return start <= cur < end
    # 跨午夜
    return cur >= start or cur < end


def resolve_item_schedule(item_spec, task_start=None, task_end=None):
    """
    场景/规则自有时段优先；否则回退任务时段。
    返回 (start, end)；皆空表示不限时段。
    """
    spec = item_spec or {}
    item_start = spec.get('schedule_start')
    item_end = spec.get('schedule_end')
    if parse_hhmm(item_start) and parse_hhmm(item_end):
        return item_start, item_end
    if parse_hhmm(task_start) and parse_hhmm(task_end):
        return task_start, task_end
    return None, None


def is_item_active_now(now, item_spec, task_start=None, task_end=None):
    start, end = resolve_item_schedule(item_spec, task_start, task_end)
    return in_daily_window(now, start, end)
