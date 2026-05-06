from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator, Set

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from camera import CameraService
from config import settings
from detector import DetectionService

app = FastAPI(title="Raspberry Pi Vision Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera = CameraService()
detector = DetectionService()
clients: Set[WebSocket] = set()
detector_task: asyncio.Task | None = None


async def detection_loop() -> None:
    last_processed = -1
    while True:
        await asyncio.sleep(0.01)
        if not camera.is_ready():
            continue

        frame_counter = camera.get_frame_counter()
        if frame_counter == last_processed:
            continue

        if frame_counter % settings.detection_interval != 0:
            continue

        frame = camera.get_frame()
        if frame is None:
            continue

        detections, annotated = detector.detect(frame)
        _ = detections
        camera.set_annotated_frame(annotated)
        last_processed = frame_counter
        await publish_updates()


async def publish_updates() -> None:
    if not clients:
        return
    payload = json.dumps(detector.get_snapshot())
    stale_clients = []
    for ws in clients:
        try:
            await ws.send_text(payload)
        except Exception:
            stale_clients.append(ws)
    for ws in stale_clients:
        clients.discard(ws)


def generate_mjpeg_stream() -> AsyncGenerator[bytes, None]:
    while True:
        frame = camera.get_annotated_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality],
        )
        if not ok:
            time.sleep(0.02)
            continue

        frame_bytes = encoded.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.03)


@app.on_event("startup")
async def startup_event() -> None:
    global detector_task
    camera.start()
    detector_task = asyncio.create_task(detection_loop())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if detector_task:
        detector_task.cancel()
    camera.stop()


@app.get("/health")
def health() -> JSONResponse:
    camera_ready = camera.is_ready()
    return JSONResponse(
        {
            "status": "ok" if camera_ready else "degraded",
            "camera_ready": camera_ready,
            "camera_error": camera.get_last_error(),
            "model_ready": detector.model_ready,
            "model_note": detector.model_note,
            "config": {
                "camera_index": settings.camera_index,
                "model_path": settings.model_path,
                "confidence_threshold": settings.confidence_threshold,
                "frame_size": [settings.frame_width, settings.frame_height],
                "detection_interval": settings.detection_interval,
            },
        }
    )


@app.get("/detections")
def get_detections() -> JSONResponse:
    return JSONResponse(detector.get_snapshot())


@app.get("/video-feed")
def video_feed() -> StreamingResponse:
    if not camera.is_ready():
        raise HTTPException(
            status_code=503,
            detail=camera.get_last_error() or "Camera stream is not ready.",
        )

    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/detections")
async def ws_detections(websocket: WebSocket) -> None:
    await websocket.accept()
    clients.add(websocket)
    try:
        await websocket.send_text(json.dumps(detector.get_snapshot()))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception:
        clients.discard(websocket)
