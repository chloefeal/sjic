"""Wake-on-LAN: 向边缘主机发送魔术包。"""
import socket


def normalize_mac(mac):
    if not mac:
        raise ValueError('MAC 地址为空')
    hexonly = ''.join(ch for ch in str(mac).upper() if ch in '0123456789ABCDEF')
    if len(hexonly) != 12:
        raise ValueError(f'无效的 MAC 地址: {mac}')
    return hexonly


def _subnet_broadcast(ip):
    parts = str(ip or '').split('.')
    if len(parts) != 4:
        return None
    if not all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
        return None
    if parts[0] == '127':
        return None
    return '.'.join(parts[:3] + ['255'])


def send_wol(mac, ip_address=None):
    """
    发送 WoL 魔术包。
    同时打全局广播、网段广播和节点上次 IP（部分网卡支持定向唤醒）。
    """
    mac_hex = normalize_mac(mac)
    mac_bytes = bytes.fromhex(mac_hex)
    packet = b'\xff' * 6 + mac_bytes * 16

    targets = [('255.255.255.255', 9), ('255.255.255.255', 7)]
    if ip_address and not str(ip_address).startswith('127.'):
        targets.append((ip_address, 9))
        targets.append((ip_address, 7))
        bcast = _subnet_broadcast(ip_address)
        if bcast:
            targets.append((bcast, 9))
            targets.append((bcast, 7))

    sent = []
    errors = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(2)
        for host, port in targets:
            try:
                sock.sendto(packet, (host, port))
                sent.append(f'{host}:{port}')
            except OSError as exc:
                errors.append(f'{host}:{port} {exc}')
    finally:
        sock.close()

    if not sent:
        raise RuntimeError('唤醒包发送失败: ' + '; '.join(errors) if errors else '无可用目标')
    return {'mac': ':'.join(mac_hex[i:i + 2] for i in range(0, 12, 2)), 'targets': sent}
