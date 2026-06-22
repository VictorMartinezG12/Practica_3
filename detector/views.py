import time

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render

from .camera import VideoCamera
from .models import DetectionEvent


def index(request):
    return render(request, 'detector/index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_feed(request):
    camera = VideoCamera()
    return StreamingHttpResponse(
        gen(camera), content_type='multipart/x-mixed-replace; boundary=frame'
    )


def latest_detections(request):
    events = DetectionEvent.objects.all()[:15]
    data = [
        {
            'objeto': e.objeto,
            'confianza': round(e.confianza, 2),
            'timestamp': e.timestamp.strftime('%H:%M:%S'),
        }
        for e in events
    ]
    return JsonResponse({'detections': data})
