from pathlib import Path
from typing import Generator

import cv2
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from stream_input import decode_mp4_with_ffmpeg

ROOT_DIR = Path(__file__).parent
VIDEO_PATH = ROOT_DIR / "traffic.mp4"
MODEL_PATH = ROOT_DIR / "yolov11n.pt"
FRAME_W, FRAME_H = 1280, 720

app = FastAPI()
app.mount("/static", StaticFiles(directory=ROOT_DIR / "static"), name="static")
templates = Jinja2Templates(directory=ROOT_DIR / "templates")

model = YOLO(str(MODEL_PATH))


def gen_frames() -> Generator[bytes, None, None]:
    for frame in decode_mp4_with_ffmpeg(str(VIDEO_PATH), FRAME_W, FRAME_H):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = model.track(bgr, persist=True, classes=[2, 5, 7], tracker="bytetrack.yaml", conf=0.1, iou=0.5, imgsz=1280)
        annotated = results[0].orig_img.copy() if hasattr(results[0], "orig_img") else bgr.copy()

        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            ids = boxes.id if boxes.id is not None else [None] * len(boxes)
            for box, tid in zip(boxes.xyxy, ids):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID {int(tid)}" if tid is not None else "car"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated, (x1, y1 - h - 4), (x1 + w + 4, y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")