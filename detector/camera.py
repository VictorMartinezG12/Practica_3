import threading
import time

import cv2

from .detection import draw_boxes, run_inference

# Correr YOLO en cada frame satura la CPU y genera lag en el stream.
# Se infiere cada N frames y se reutilizan las últimas cajas detectadas
# en los frames intermedios, manteniendo el video fluido.
DETECT_EVERY_N_FRAMES = 3
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80


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
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        last_boxes = []
        frame_count = 0
        while self.running:
            success, frame = self.video.read()
            if not success:
                time.sleep(0.1)
                continue

            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                last_boxes = run_inference(frame)
            frame_count += 1

            frame = draw_boxes(frame, last_boxes)
            ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with self.frame_lock:
                    self.frame = jpeg.tobytes()

    def get_frame(self):
        with self.frame_lock:
            return self.frame
