import time
from threading import Lock

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .models import DetectionEvent

_coco_model = None
_helmet_model = None
_model_lock = Lock()

COCO_TARGET_CLASSES = {
    'cell phone', 'person', 'backpack', 'bottle', 'laptop', 'book', 'cup',
}

# Modelo público (Hugging Face) entrenado para detectar equipo de protección,
# incluye casco de moto/ciclista ('helmet') y su ausencia ('no_helmet').
# COCO no tiene clase "casco", por eso se usa un segundo modelo especializado.
HELMET_MODEL_REPO = 'keremberke/yolov8n-protective-equipment-detection'
HELMET_MODEL_FILE = 'best.pt'
HELMET_TARGET_CLASSES = {'helmet', 'no_helmet'}

BOX_COLORS = {
    'no_helmet': (0, 0, 255),  # rojo: sin casco
}
DEFAULT_COLOR = (0, 255, 0)  # verde: resto de clases

_last_saved = {}
_THROTTLE_SECONDS = 1.0

# Tamaño reducido para la inferencia: acelera YOLO en CPU sin perder
# demasiada precisión para objetos cercanos a la cámara.
INFERENCE_SIZE = 320


def get_coco_model():
    global _coco_model
    with _model_lock:
        if _coco_model is None:
            _coco_model = YOLO('yolov8n.pt')
    return _coco_model


def get_helmet_model():
    global _helmet_model
    with _model_lock:
        if _helmet_model is None:
            weights_path = hf_hub_download(repo_id=HELMET_MODEL_REPO, filename=HELMET_MODEL_FILE)
            _helmet_model = YOLO(weights_path)
    return _helmet_model


def _maybe_save_event(label, confidence):
    now = time.time()
    last = _last_saved.get(label, 0)
    if now - last >= _THROTTLE_SECONDS:
        _last_saved[label] = now
        DetectionEvent.objects.create(objeto=label, confianza=confidence)


def _run(model, frame, target_classes):
    results = model.predict(frame, imgsz=INFERENCE_SIZE, verbose=False)[0]

    boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label not in target_classes:
            continue
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append((label, confidence, x1, y1, x2, y2))
        _maybe_save_event(label, confidence)

    return boxes


def run_inference_coco(frame):
    """Detección de objetos generales (celulares, personas, mochilas, etc.)."""
    return _run(get_coco_model(), frame, COCO_TARGET_CLASSES)


def run_inference_helmet(frame):
    """Detección de casco de moto/ciclista (con/sin casco)."""
    return _run(get_helmet_model(), frame, HELMET_TARGET_CLASSES)


def draw_boxes(frame, boxes):
    for label, confidence, x1, y1, x2, y2 in boxes:
        color = BOX_COLORS.get(label, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confidence:.2f}"
        cv2.putText(frame, text, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
