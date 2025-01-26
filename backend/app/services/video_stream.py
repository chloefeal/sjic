from flask_socketio import emit
import cv2
import numpy as np

class VideoStreamService:
    def __init__(self):
        self.active_streams = {}
    
    def start_stream(self, camera_id, rtsp_url):
        """启动视频流"""
        if camera_id not in self.active_streams:
            cap = cv2.VideoCapture(rtsp_url)
            self.active_streams[camera_id] = {
                'capture': cap,
                'active': True
            }
            return True
        return False

    def stop_stream(self, camera_id):
        """停止视频流"""
        if camera_id in self.active_streams:
            self.active_streams[camera_id]['active'] = False
            self.active_streams[camera_id]['capture'].release()
            del self.active_streams[camera_id]
            return True
        return False 