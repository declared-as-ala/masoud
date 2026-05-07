# Raspberry Pi + React Vegetable Quality Dashboard

This project has two parts:

- `raspberry-backend` (runs on Raspberry Pi, handles camera + AI + APIs)
- `react-dashboard` (runs on PC, shows live stream + detections dashboard)

## 1) Raspberry Pi setup

### Install dependencies

```bash
cd raspberry-backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Connect the camera

- USB camera: plug in and use default `CAMERA_INDEX=0`.
- Raspberry Pi Camera Module: enable camera in Raspberry Pi configuration and test with OpenCV.

If camera is not found, check:
- physical connection
- camera permissions
- correct `CAMERA_INDEX` value

### Use OpenCV pretrained mode (no training)

By default, this backend now uses:
- OpenCV MobileNet-SSD pretrained model for **person detection**
- OpenCV HSV/contour heuristics for **tomato/pepper good/bad** (no training)

Set mode with env var (optional, default already set):

```bash
export DETECTOR_BACKEND=opencv_pretrained
```

On first run, MobileNet-SSD files are auto-downloaded into:
- `./models/opencv/MobileNetSSD_deploy.prototxt`
- `./models/opencv/MobileNetSSD_deploy.caffemodel`

You can change folder with:

```bash
export OPENCV_MODEL_DIR=./models/opencv
```

### Optional: custom AI model mode

Create a folder and place your model file:

```bash
mkdir -p models
```

Supported path examples:
- `./models/vegetable_quality.pt`
- `./models/vegetable_quality.onnx`
- `./models/vegetable_quality.tflite` (placeholder integration point in `detector.py`)

Set with env var:

```bash
export DETECTOR_BACKEND=custom_model
export MODEL_PATH=./models/vegetable_quality.pt
```

Your model should output classes:
- `tomato_good`
- `tomato_bad`
- `pepper_good`
- `pepper_bad`
- `person`

### Run FastAPI backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /health`
- `GET /video-feed` (MJPEG stream)
- `GET /detections`
- `WS /ws/detections`

## 2) React dashboard setup (on PC)

```bash
cd react-dashboard
npm install
npm run dev
```

Create `.env` in `react-dashboard`:

```bash
VITE_RPI_API_URL=http://RASPBERRY_PI_IP:8000
```

## 3) Find Raspberry Pi IP address

On Raspberry Pi:

```bash
hostname -I
```

Use the shown IP in dashboard `.env`.

## 4) Config variables

Backend config is in `config.py` and reads env vars:

- `CAMERA_INDEX` (default `0`)
- `DETECTOR_BACKEND` (default `opencv_pretrained`, or `custom_model`)
- `MODEL_PATH` (default `./models/vegetable_quality.pt`, used in `custom_model` mode)
- `OPENCV_MODEL_DIR` (default `./models/opencv`)
- `CONFIDENCE_THRESHOLD` (default `0.4`)
- `FRAME_WIDTH` (default `640`)
- `FRAME_HEIGHT` (default `480`)
- `DETECTION_INTERVAL` (default `3`) -> runs detection every N frames
- `JPEG_QUALITY` (default `80`)

## 5) Replace or train model later

1. Train a custom model (YOLOv8/TFLite) for your required classes.
2. Copy model file to Raspberry Pi (for example into `raspberry-backend/models/`).
3. Update `MODEL_PATH`.
4. Restart backend.

For TFLite:
- `detector.py` already contains a clean placeholder branch for `.tflite`.
- Add your `tflite-runtime` inference logic in that branch when your model is ready.

## 6) Performance notes (Raspberry Pi)

- Use lightweight model (`yolov8n` or quantized TFLite).
- Keep frame size small (`640x480` or lower).
- Increase `DETECTION_INTERVAL` for smoother stream.
- Use hardware-accelerated camera path if available on your Pi model.
