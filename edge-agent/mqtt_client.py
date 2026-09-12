import paho.mqtt.client as mqtt
import json
import logging
import os
import threading
import time
from engine.task_manager import TaskManager

logger = logging.getLogger(__name__)

class EdgeMqttClient:
    def __init__(self, config, task_manager: TaskManager):
        self.config = config
        self.edge_id = config['edge_id']
        self.broker = config['mqtt']['broker_url']
        self.port = config['mqtt']['broker_port']
        self.client = mqtt.Client(client_id=f"sjic-edge-{self.edge_id}")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        self.task_manager = task_manager
        self._restart_lock = threading.Lock()

        self.topic_prefix = f"sjic/edge/{self.edge_id}"

        # 设置“遗嘱”消息 (LWT): 如果盒子异常断开，Broker 会自动发这条消息告知后端
        self.client.will_set(
            topic=f"{self.topic_prefix}/heartbeat",
            payload=json.dumps({
                "edge_id": self.edge_id,
                "status": "offline",
                "message": "MQTT connection lost (LWT triggered)"
            }),
            qos=1,
            retain=True
        )

    def start(self):
        try:
            self.client.connect(self.broker, self.port, self.config['mqtt']['keepalive'])
            self.client.loop_start()
            logger.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {str(e)}")

    def stop(self):
        try:
            # 优雅退出时主动向云端通告 offline 状态，并更新 Broker 保留消息
            offline_payload = {
                "edge_id": self.edge_id,
                "status": "offline",
                "message": "Edge agent shut down gracefully"
            }
            self.client.publish(f"{self.topic_prefix}/heartbeat", json.dumps(offline_payload), qos=1, retain=True)
        except Exception as e:
            logger.warning(f"Failed to publish offline status on stop: {e}")
        self.client.loop_stop()
        self.client.disconnect()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to Platform successfully.")
            # 订阅与自己相关的指令
            client.subscribe(f"{self.topic_prefix}/task/start")
            client.subscribe(f"{self.topic_prefix}/task/stop")
            client.subscribe(f"{self.topic_prefix}/agent/restart")
        else:
            logger.error(f"Failed to connect, return code {rc}")

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection (rc={rc}). Client will auto-reconnect.")

    def on_message(self, client, userdata, msg):
        payload_str = msg.payload.decode('utf-8')
        logger.info(f"Received message on topic {msg.topic}")
        try:
            data = json.loads(payload_str) if payload_str.strip() else {}
            if msg.topic.endswith("/task/start"):
                self.task_manager.start_task(data)
            elif msg.topic.endswith("/task/stop"):
                task_id = data.get("task_id")
                if task_id:
                    self.task_manager.stop_task(task_id)
            elif msg.topic.endswith("/agent/restart"):
                self._handle_agent_restart(data)
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")

    def _handle_agent_restart(self, data):
        if not self._restart_lock.acquire(blocking=False):
            logger.warning("Agent restart already in progress, ignoring duplicate command")
            return

        reason = (data or {}).get("reason", "admin")
        logger.warning(f"Agent restart requested via MQTT, reason={reason}")

        def _restart():
            try:
                self.task_manager.stop_all_tasks_for_restart(join_timeout=8)
            except Exception as e:
                logger.error(f"Error stopping tasks before restart: {e}")
            try:
                self.stop()
            except Exception as e:
                logger.warning(f"Error during MQTT stop before restart: {e}")
            time.sleep(0.3)
            logger.warning("Exiting edge agent process for Docker Compose restart")
            os._exit(0)

        threading.Thread(target=_restart, name="agent-restart", daemon=True).start()

    def publish_heartbeat(self, status_payload):
        topic = f"{self.topic_prefix}/heartbeat"
        # 使用 retain=True 覆盖 Broker 中可能驻留的旧 LWT 离线消息
        self.client.publish(topic, json.dumps(status_payload), qos=1, retain=True)
        
    def publish_task_status(self, payload):
        topic = f"{self.topic_prefix}/task/status"
        self.client.publish(topic, json.dumps(payload), qos=1)
