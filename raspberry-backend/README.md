# Raspberry Pi Smart Camera Backend

FastAPI backend for:
- person detection using pretrained YOLO (`yolov8n.pt`)
- tomato/pepper detection from pretrained custom model (`models/vegetables.pt`)
- face authorization using local images in `faces_authorized/`
- Telegram unauthorized alerts with cooldown
- MJPEG live stream + REST endpoints for dashboard

## 1) Install Raspberry Pi dependencies

```bash
cd raspberry-backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Enable/connect camera

- USB camera: use default `CAMERA_INDEX=0`.
- Raspberry Pi Camera Module: enable camera in Pi settings and reboot.
- Test camera with OpenCV if needed.

If camera fails, `/health` returns `camera_ready: false` and an error message.

## 3) Add authorized faces

Put clear images in:
- `faces_authorized/ala.jpg`
- `faces_authorized/technician.jpg`

File name becomes the authorized person name on dashboard.

## 4) Add pretrained vegetable model

Put your pretrained model at:
- `models/vegetables.pt`

Required classes:
- `tomato`
- `pepper` (or `felfel`)

If file is missing, backend still works and shows warning:
- `Vegetable model not found. Place your pretrained model at models/vegetables.pt`

## 5) Configure Telegram

Edit hardcoded values in `config.py`:
- `telegram_bot_token`
- `telegram_chat_id`
- `alert_cooldown_seconds`

Unauthorized detection sends photo + timestamp + status.

## 6) Run backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `GET /video-feed`
- `GET /detections`
- `GET /alerts`

## 7) Run React dashboard

```bash
cd ../react-dashboard
npm install
npm run dev
```

## 8) Find Raspberry Pi IP

```bash
hostname -I
```

Use that IP in dashboard `.env`.

## 9) Test stream in browser

Open:
- `http://RASPBERRY_PI_IP:8000/video-feed`

## 10) Improve performance on weak Raspberry Pi

- keep frame size at `640x480` or lower
- increase `DETECTION_INTERVAL_FRAMES` (e.g. `6` or `8`)
- keep `PERSON_MODEL=yolov8n.pt`
- disable face auth temporarily with `FACE_AUTH_ENABLED=false`
- increase `FACE_RECHECK_INTERVAL_FRAMES` to reduce recognition load
