# Detección de Objetos en Tiempo Real con Django y OpenCV

Aplicativo web que captura video en tiempo real desde la cámara, lo procesa con OpenCV y detecta objetos (celulares, personas, laptops, mochilas, botellas, libros, tazas) y casco de moto/ciclista (con/sin casco) usando modelos YOLOv8 preentrenados, mostrando los resultados directamente sobre el video en una interfaz web.

## Requisitos

- Python 3.10+
- PostgreSQL
- Una cámara web

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd PRACTICO_EXPERIMENTAL_3
   ```

2. Crear y activar un entorno virtual:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Crear la base de datos en PostgreSQL:
   ```bash
   sudo -u postgres createdb deteccion_cv
   ```

5. Crear un archivo `.env` en la raíz del proyecto (basado en `.env.example`):
   ```
   SECRET_KEY=tu-clave-secreta
   DEBUG=True
   DB_NAME=deteccion_cv
   DB_USER=postgres
   DB_PASSWORD=tu-contraseña
   DB_HOST=localhost
   DB_PORT=5432
   ```

6. Aplicar migraciones:
   ```bash
   python manage.py migrate
   ```

7. Ejecutar el servidor de desarrollo:
   ```bash
   python manage.py runserver
   ```

8. Abrir `http://localhost:8000/` en el navegador. La primera vez se descargarán automáticamente los modelos `yolov8n.pt` (Ultralytics) y `keremberke/yolov8n-protective-equipment-detection` (Hugging Face).

## Funcionalidad

- **Streaming de video en tiempo real** (`/video_feed/`) usando `cv2.VideoCapture` y `StreamingHttpResponse` (MJPEG).
- **Detección de objetos** con YOLOv8 (Ultralytics), entrenado sobre el dataset COCO: detecta celulares, personas, laptops, mochilas, botellas, libros y tazas, dibujando el cuadro delimitador y la confianza sobre el video.
- **Detección de casco de moto/ciclista** con un segundo modelo YOLOv8 especializado ([keremberke/yolov8n-protective-equipment-detection](https://huggingface.co/keremberke/yolov8n-protective-equipment-detection)), ya que el dataset COCO no incluye esa clase. Dibuja el cuadro en verde si detecta `helmet` (con casco) y en rojo si detecta `no_helmet` (sin casco).
- Los dos modelos se alternan cada pocos frames para mantener el video fluido sin saturar la CPU.
- **Registro de detecciones** en PostgreSQL (`detector.DetectionEvent`), con un panel en la interfaz que muestra las últimas detecciones en tiempo real.

## Estructura del proyecto

```
config/             # Configuración del proyecto Django
detector/           # App principal: captura, detección y vistas
  camera.py          # Captura de video con OpenCV en un hilo
  detection.py        # Carga del modelo YOLOv8 y dibujo de detecciones
  models.py            # Modelo DetectionEvent (PostgreSQL)
  templates/detector/  # Interfaz web
```

## Entregables

- Video demostrativo (máx. 3 min) mostrando la detección en tiempo real.
- Repositorio GitHub con `README.md` y `requirements.txt` (sin entorno virtual).
