import time
from threading import Lock

import cv2
from ultralytics import YOLO

from .models import DetectionEvent

_model = None
_model_lock = Lock()

TARGET_CLASSES = {
    'cell phone', 'person', 'backpack', 'bottle', 'laptop', 'book', 'cup',
}

_last_saved = {}
_THROTTLE_SECONDS = 1.0


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = YOLO('yolov8n.pt')
    return _model


def _maybe_save_event(label, confidence):
    now = time.time()
    last = _last_saved.get(label, 0)
    if now - last >= _THROTTLE_SECONDS:
        _last_saved[label] = now
        DetectionEvent.objects.create(objeto=label, confianza=confidence)


def detect_objects(frame):
    model = get_model()
    results = model.predict(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label not in TARGET_CLASSES:
            continue
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {confidence:.2f}"
        cv2.putText(frame, text, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _maybe_save_event(label, confidence)

    return frame
