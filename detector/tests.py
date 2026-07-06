"""
Suite de pruebas del proyecto.

- DetectionUnitTests       -> Pruebas unitarias del módulo detector/detection.py
- CameraUnitTests          -> Pruebas unitarias del módulo detector/camera.py
- ViewsIntegrationTests    -> Pruebas de integración de las rutas Django
- FunctionalFlowTests      -> Pruebas funcionales de los casos de uso clave

Los modelos YOLO reales (COCO y casco) y la cámara física se sustituyen por
dobles de prueba (fakes/mocks) para que la suite sea determinista, rápida y
ejecutable en cualquier entorno, sin depender de hardware ni de descargar
pesos de modelos.
"""
import time
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import Client, TestCase
from django.urls import reverse

from . import detection
from .models import DetectionEvent


# ---------------------------------------------------------------------------
# Dobles de prueba (fakes) para no depender de los modelos YOLO reales
# ---------------------------------------------------------------------------

class FakeBox:
    """Imita la interfaz de ultralytics.engine.results.Boxes para una sola caja."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class FakeResults:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Sustituye a un modelo YOLO cargado: mismo contrato (.names, .predict())."""

    def __init__(self, names, boxes):
        self.names = names
        self._boxes = boxes
        self.predict_calls = []

    def predict(self, frame, imgsz=None, verbose=False):
        self.predict_calls.append({'imgsz': imgsz, 'verbose': verbose})
        return [FakeResults(self._boxes)]


# ---------------------------------------------------------------------------
# Sesión 1 - Pruebas unitarias: detector/detection.py
# ---------------------------------------------------------------------------

class DetectionUnitTests(TestCase):
    """Pruebas unitarias sobre las funciones críticas de detección (OpenCV/YOLO)."""

    def setUp(self):
        detection._last_saved.clear()
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_run_filtra_solo_las_clases_objetivo(self):
        """_run debe descartar detecciones cuya clase no esté en target_classes."""
        names = {0: 'person', 1: 'dog', 2: 'cell phone'}
        boxes = [
            FakeBox(0, 0.90, (10, 10, 50, 50)),   # person -> se conserva
            FakeBox(1, 0.80, (5, 5, 20, 20)),      # dog -> se descarta
            FakeBox(2, 0.70, (30, 30, 60, 60)),    # cell phone -> se conserva
        ]
        model = FakeModel(names, boxes)

        result = detection._run(model, self.frame, {'person', 'cell phone'})

        labels = {label for label, *_ in result}
        self.assertEqual(labels, {'person', 'cell phone'})
        self.assertEqual(len(result), 2)

    def test_run_devuelve_coordenadas_y_confianza_correctas(self):
        names = {0: 'person'}
        boxes = [FakeBox(0, 0.8734, (1, 2, 3, 4))]
        model = FakeModel(names, boxes)

        result = detection._run(model, self.frame, {'person'})

        self.assertEqual(result, [('person', 0.8734, 1, 2, 3, 4)])

    def test_run_sin_detecciones_devuelve_lista_vacia(self):
        model = FakeModel({0: 'person'}, [])
        result = detection._run(model, self.frame, {'person'})
        self.assertEqual(result, [])

    def test_run_invoca_predict_con_tamano_de_inferencia_reducido(self):
        """El tamaño de imagen usado en predict debe ser el configurado (rendimiento)."""
        model = FakeModel({0: 'person'}, [])
        detection._run(model, self.frame, {'person'})
        self.assertEqual(model.predict_calls[0]['imgsz'], detection.INFERENCE_SIZE)

    def test_run_inference_coco_usa_clases_coco(self):
        names = {0: 'person', 1: 'dog'}
        boxes = [FakeBox(0, 0.9, (0, 0, 10, 10)), FakeBox(1, 0.9, (0, 0, 10, 10))]
        fake_model = FakeModel(names, boxes)
        with patch.object(detection, 'get_coco_model', return_value=fake_model):
            result = detection.run_inference_coco(self.frame)
        self.assertEqual([label for label, *_ in result], ['person'])

    def test_run_inference_helmet_usa_clases_de_casco(self):
        names = {0: 'helmet', 1: 'no_helmet', 2: 'person'}
        boxes = [
            FakeBox(0, 0.95, (0, 0, 10, 10)),
            FakeBox(1, 0.88, (0, 0, 10, 10)),
            FakeBox(2, 0.5, (0, 0, 10, 10)),
        ]
        fake_model = FakeModel(names, boxes)
        with patch.object(detection, 'get_helmet_model', return_value=fake_model):
            result = detection.run_inference_helmet(self.frame)
        labels = {label for label, *_ in result}
        self.assertEqual(labels, {'helmet', 'no_helmet'})

    def test_maybe_save_event_crea_un_registro(self):
        detection._maybe_save_event('person', 0.91)
        self.assertEqual(DetectionEvent.objects.count(), 1)
        evento = DetectionEvent.objects.first()
        self.assertEqual(evento.objeto, 'person')
        self.assertAlmostEqual(evento.confianza, 0.91)

    def test_maybe_save_event_aplica_throttle(self):
        """Dos eventos del mismo objeto dentro de la ventana de throttle -> 1 solo registro."""
        detection._maybe_save_event('person', 0.9)
        detection._maybe_save_event('person', 0.95)
        self.assertEqual(DetectionEvent.objects.count(), 1)

    def test_maybe_save_event_permite_nuevo_registro_tras_throttle(self):
        with patch.object(detection.time, 'time', return_value=1000.0):
            detection._maybe_save_event('person', 0.9)
        with patch.object(detection.time, 'time', return_value=1000.0 + detection._THROTTLE_SECONDS + 0.1):
            detection._maybe_save_event('person', 0.92)
        self.assertEqual(DetectionEvent.objects.count(), 2)

    def test_maybe_save_event_no_comparte_throttle_entre_objetos(self):
        detection._maybe_save_event('person', 0.9)
        detection._maybe_save_event('cell phone', 0.8)
        self.assertEqual(DetectionEvent.objects.count(), 2)

    def test_run_dispara_guardado_de_evento_por_deteccion(self):
        model = FakeModel({0: 'person'}, [FakeBox(0, 0.9, (0, 0, 10, 10))])
        detection._run(model, self.frame, {'person'})
        self.assertEqual(DetectionEvent.objects.count(), 1)

    def test_draw_boxes_no_falla_y_conserva_dimensiones(self):
        boxes = [('person', 0.87, 10, 10, 50, 50)]
        frame = detection.draw_boxes(self.frame.copy(), boxes)
        self.assertEqual(frame.shape, self.frame.shape)
        # El rectángulo dibujado modifica píxeles que antes eran cero.
        self.assertTrue((frame != 0).any())

    def test_draw_boxes_usa_rojo_para_no_helmet(self):
        boxes = [('no_helmet', 0.9, 10, 10, 50, 50)]
        frame = detection.draw_boxes(self.frame.copy(), boxes)
        # El borde superior del rectángulo debe llevar el color rojo (BGR) configurado.
        pixel = frame[10, 30]
        self.assertEqual(tuple(int(c) for c in pixel), detection.BOX_COLORS['no_helmet'])

    def test_draw_boxes_usa_verde_por_defecto(self):
        boxes = [('person', 0.9, 10, 10, 50, 50)]
        frame = detection.draw_boxes(self.frame.copy(), boxes)
        pixel = frame[10, 30]
        self.assertEqual(tuple(int(c) for c in pixel), detection.DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Sesión 1 - Pruebas unitarias: detector/camera.py
# ---------------------------------------------------------------------------

class CameraUnitTests(TestCase):
    """Pruebas unitarias de la lógica de alternancia/throttle de inferencia por frame."""

    def _make_camera_without_thread(self):
        """Crea una instancia de VideoCamera sin arrancar el hilo ni tocar hardware real."""
        from detector import camera as camera_module

        camera_module.VideoCamera._instance = None
        with patch.object(camera_module.cv2, 'VideoCapture') as mock_capture, \
             patch.object(camera_module.threading.Thread, 'start'):
            mock_capture.return_value = MagicMock()
            cam = camera_module.VideoCamera()
        return cam, camera_module

    def tearDown(self):
        from detector import camera as camera_module
        camera_module.VideoCamera._instance = None

    def test_es_singleton(self):
        cam1, camera_module = self._make_camera_without_thread()
        with patch.object(camera_module.cv2, 'VideoCapture'), \
             patch.object(camera_module.threading.Thread, 'start'):
            cam2 = camera_module.VideoCamera()
        self.assertIs(cam1, cam2)

    def test_get_frame_none_antes_de_procesar(self):
        cam, _ = self._make_camera_without_thread()
        self.assertIsNone(cam.get_frame())

    def test_alterna_coco_y_casco_cada_n_frames(self):
        """Cada DETECT_EVERY_N_FRAMES se debe alternar entre modelo COCO y de casco."""
        from detector import camera as camera_module

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cam, _ = self._make_camera_without_thread()
        cam.running = True

        call_order = []

        def fake_coco(_frame):
            call_order.append('coco')
            return []

        def fake_helmet(_frame):
            call_order.append('helmet')
            return []

        reads = [(True, frame)] * (camera_module.DETECT_EVERY_N_FRAMES * 4)

        def stop_after_reads(*_args, **_kwargs):
            if reads:
                return reads.pop(0)
            cam.running = False
            return (False, None)

        cam.video.read.side_effect = stop_after_reads

        with patch.object(camera_module, 'run_inference_coco', side_effect=fake_coco), \
             patch.object(camera_module, 'run_inference_helmet', side_effect=fake_helmet):
            cam._update()

        self.assertEqual(call_order, ['coco', 'helmet', 'coco', 'helmet'])

    def test_get_frame_devuelve_jpeg_tras_procesar(self):
        from detector import camera as camera_module

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cam, _ = self._make_camera_without_thread()
        cam.video.read.side_effect = [(True, frame), (False, None)]
        cam.running = True

        def stop(*_a, **_k):
            cam.running = False
            return []

        with patch.object(camera_module, 'run_inference_coco', side_effect=stop), \
             patch.object(camera_module, 'run_inference_helmet', return_value=[]):
            cam._update()

        self.assertIsNotNone(cam.get_frame())
        self.assertIsInstance(cam.get_frame(), bytes)


# ---------------------------------------------------------------------------
# Sesión 1 - Pruebas de integración: rutas Django
# ---------------------------------------------------------------------------

class FakeCamera:
    """Doble de VideoCamera para las pruebas de integración/funcionales de las vistas."""

    FRAME = b'--fake-jpeg-bytes--'

    def get_frame(self):
        return self.FRAME


class ViewsIntegrationTests(TestCase):
    """Pruebas de integración: validan el flujo HTTP a través de las rutas principales."""

    def setUp(self):
        self.client = Client()

    def test_index_responde_200_y_usa_el_template_esperado(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'detector/index.html')

    def test_index_incluye_el_tag_de_video(self):
        response = self.client.get(reverse('index'))
        self.assertContains(response, 'id="video-feed"')
        self.assertContains(response, reverse('video_feed'))

    @patch('detector.views.VideoCamera', return_value=FakeCamera())
    def test_video_feed_responde_stream_multipart(self, _mock_camera):
        response = self.client.get(reverse('video_feed'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('multipart/x-mixed-replace', response['Content-Type'])
        primer_chunk = next(iter(response.streaming_content))
        self.assertIn(b'Content-Type: image/jpeg', primer_chunk)
        self.assertIn(FakeCamera.FRAME, primer_chunk)

    def test_latest_detections_devuelve_json_vacio_sin_eventos(self):
        response = self.client.get(reverse('latest_detections'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'detections': []})

    def test_latest_detections_devuelve_los_eventos_mas_recientes_primero(self):
        DetectionEvent.objects.create(objeto='person', confianza=0.9111)
        DetectionEvent.objects.create(objeto='cell phone', confianza=0.8222)

        response = self.client.get(reverse('latest_detections'))
        data = response.json()['detections']

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['objeto'], 'cell phone')  # más reciente primero
        self.assertEqual(data[1]['objeto'], 'person')
        self.assertEqual(data[0]['confianza'], 0.82)  # redondeado a 2 decimales

    def test_latest_detections_limita_a_15_resultados(self):
        for i in range(20):
            DetectionEvent.objects.create(objeto=f'obj{i}', confianza=0.5)

        response = self.client.get(reverse('latest_detections'))
        self.assertEqual(len(response.json()['detections']), 15)


# ---------------------------------------------------------------------------
# Sesión 2 - Pruebas funcionales: casos de uso clave
# ---------------------------------------------------------------------------

class FunctionalFlowTests(TestCase):
    """
    Simulan el flujo real de un usuario:
    1) abre la aplicación (inicio),
    2) el video se detecta y transmite (detección en video),
    3) la interfaz visualiza las detecciones (visualización en interfaz).
    """

    def setUp(self):
        self.client = Client()

    def test_caso_de_uso_inicio_de_la_aplicacion(self):
        """El usuario abre la app y recibe la página principal con el stream enlazado."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Detección de Objetos en Tiempo Real')

    @patch('detector.views.VideoCamera', return_value=FakeCamera())
    def test_caso_de_uso_deteccion_en_video(self, _mock_camera):
        """El usuario recibe frames del video con las detecciones dibujadas."""
        response = self.client.get('/video_feed/')
        chunk = next(iter(response.streaming_content))
        self.assertIn(b'--frame', chunk)
        self.assertIn(b'image/jpeg', chunk)

    def test_caso_de_uso_visualizacion_de_detecciones_en_interfaz(self):
        """Tras generarse detecciones, la interfaz debe poder listarlas vía la API."""
        # Simula que el módulo de detección guardó eventos reales.
        detection._last_saved.clear()
        detection._maybe_save_event('person', 0.93)
        detection._maybe_save_event('no_helmet', 0.77)

        response = self.client.get('/latest_detections/')
        objetos = {d['objeto'] for d in response.json()['detections']}

        self.assertEqual(response.status_code, 200)
        self.assertEqual(objetos, {'person', 'no_helmet'})

    @patch('detector.views.VideoCamera', return_value=FakeCamera())
    def test_flujo_completo_de_uso(self, _mock_camera):
        """Recorre inicio -> stream de video -> panel de detecciones, en una sola sesión."""
        inicio = self.client.get('/')
        self.assertEqual(inicio.status_code, 200)

        video = self.client.get('/video_feed/')
        self.assertEqual(video.status_code, 200)
        next(iter(video.streaming_content))

        DetectionEvent.objects.create(objeto='laptop', confianza=0.6)
        detecciones = self.client.get('/latest_detections/')
        self.assertEqual(detecciones.status_code, 200)
        self.assertEqual(detecciones.json()['detections'][0]['objeto'], 'laptop')
