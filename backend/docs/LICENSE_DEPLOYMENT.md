# 授权部署说明

## 行为

1. **未导入授权**：可用默认账号登录，但业务功能全部不可用，需先导入授权文件。
2. **试用版 (`trial`)**：自签发日起 **30 天**，并限制 `max_cameras`（视频接入路数）。
3. **正式版 (`official`)**：按签发时填写的 `expires_at` 与 `max_cameras` 生效。
4. 授权文件带 **HMAC 签名**，并绑定本机 `machine_code`，防止篡改与虚机镜像复制后直接复用。

授权文件默认路径：`backend/instance/license.json`（Docker 已挂载 `instance/`）。

## 机器码（防克隆）

`machine_code` = `SHA256`（下列因子按 key 排序后拼接），**不含容器 hostname**。

| 因子 | 来源 | 说明 |
|------|------|------|
| `machine_id` | `/etc/machine-id`（或 `/var/lib/dbus/machine-id`） | 宿主机 machine-id；Docker 建议挂载宿主机文件 |
| `mac_node` | `uuid.getnode()` | 本机网卡 MAC（48-bit 整数） |
| `product_uuid` | `/sys/class/dmi/id/product_uuid` | 主板/虚机 UUID（有则加入） |
| `board_serial` / `product_serial` | DMI serial | 有真实值才加入（过滤 OEM 占位） |
| `windows_uuid` | Win32 `ComputerSystemProduct.UUID` | 仅 Windows |
| `system` / `release` / `machine` | `platform.*` | OS 族、版本、架构 |

克隆虚机后若 `machine-id` / 虚拟机 UUID / MAC 任一变化，机器码即变，旧授权失效。

Docker 部署请将宿主机 machine-id 挂入容器（在根目录 `docker-compose.yml` 中取消注释）：

```yaml
- /etc/machine-id:/etc/machine-id:ro
```

Windows 本机开发可不挂载；依赖 Windows 机器 UUID + MAC 计算机器码。

## 签发授权

在发行方机器上（需与线上一致的 `license.signing_key`）：

```bash
cd backend

# 试用版（自动 30 天）
python scripts/generate_license.py \
  --machine-code <客户机器码> \
  --edition trial \
  --max-cameras 4 \
  --out license-trial.json

# 正式版
python scripts/generate_license.py \
  --machine-code <客户机器码> \
  --edition official \
  --max-cameras 32 \
  --expires-at 2027-12-31 \
  --customer-id acme \
  --out license-official.json
```

`issued_at` / `expires_at` 使用 **YYYY-MM-DD**（到期日当天仍有效）。

客户从「系统设置 → 授权管理」复制机器码并发给发行方；拿到文件后在同一页面导入。

也可调用 API：

- `GET /api/license/status`
- `POST /api/license/import`（multipart 字段 `license`，或 JSON body）

## 配置

`backend/config.yaml`：

```yaml
license:
  trial_max_cameras: 4          # 仅作签发参考/状态展示
  signing_key: "change-me"      # 须与 generate_license.py 一致；也可用环境变量 LICENSE_SIGNING_KEY
```

## 运行时约束

- 全局：无有效授权时，除登录 / 品牌 / 授权接口外，业务 API 返回 403。
- 创建摄像头时校验视频路数配额。
- 任务创建/更新/启动校验算法授权（`allowed_algorithms` 为空表示不限制算法）。
