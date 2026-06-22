import threading
import time

import cv2

from .detection import detect_objects


class VideoCamera:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.video = cv2.VideoCapture(0)
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            success, frame = self.video.read()
            if not success:
                time.sleep(0.1)
                continue
            frame = detect_objects(frame)
            ok, jpeg = cv2.imencode('.jpg', frame)
            if ok:
                with self.frame_lock:
                    self.frame = jpeg.tobytes()
            time.sleep(0.03)

    def get_frame(self):
        with self.frame_lock:
            return self.frame
