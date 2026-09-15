"""在边缘容器内对宿主机执行重启 / 关机。"""
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def _run(cmd, timeout=15):
    logger.warning(f"Host power command: {cmd}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _try_commands(candidates):
    last_err = None
    for cmd in candidates:
        bin_name = cmd[0]
        if os.path.isabs(bin_name):
            if not os.path.exists(bin_name):
                continue
        elif not shutil.which(bin_name):
            continue
        try:
            result = _run(cmd)
            if result.returncode == 0:
                return True, None
            last_err = (result.stderr or result.stdout or f'exit {result.returncode}').strip()
            logger.warning(f"Command failed ({cmd}): {last_err}")
        except FileNotFoundError:
            continue
        except Exception as exc:
            last_err = str(exc)
            logger.warning(f"Command error ({cmd}): {last_err}")
    return False, last_err


def run_host_power(action):
    """
    action: reboot | shutdown
    优先 nsenter 进入宿主机 PID 1（需 compose: privileged + pid: host）。
    """
    if action not in ('reboot', 'shutdown'):
        raise ValueError(f'unsupported power action: {action}')

    if os.name == 'nt':
        flag = '/r' if action == 'reboot' else '/s'
        ok, err = _try_commands([['shutdown', flag, '/t', '0']])
        if not ok:
            raise RuntimeError(err or 'Windows shutdown failed')
        return

    host_cmd = ['reboot'] if action == 'reboot' else ['poweroff']
    alt_cmd = ['shutdown', '-r', 'now'] if action == 'reboot' else ['shutdown', '-h', 'now']
    nsenter = ['nsenter', '-t', '1', '-m', '-u', '-i', '-n', '-p', '--']

    candidates = [
        nsenter + host_cmd,
        nsenter + alt_cmd,
        host_cmd,
        alt_cmd,
        ['systemctl', action if action == 'reboot' else 'poweroff'],
    ]
    ok, err = _try_commands(candidates)
    if not ok:
        raise RuntimeError(
            err or '无法执行主机电源指令。请确认边缘 Docker 使用 privileged 与 pid: host。'
        )
