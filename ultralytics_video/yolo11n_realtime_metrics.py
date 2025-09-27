import argparse
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Realtime YOLOv11 Pose (yolo11n-pose.pt) runner with on-screen metrics.
# - Supports webcam (default) or a video file path via --source
# - Overlays detection and runtime metrics (inference time, FPS, RT factor, people count)
# - Keys: q (quit), p (pause), s (toggle sync-to-fps)


def format_ms(ms: float) -> str:
    return f"{ms:.1f} ms"


def draw_text(img, text, x, y, scale=0.6, color=(255, 255, 255), thickness=1):
    # Black outline then colored text for readability
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def guess_default_model(script_dir: Path) -> str:
    # Try common locations for yolo11n-pose.pt
    candidates = [
        script_dir / "yolo11n-pose.pt",
        script_dir.parent / "hockeypose_1" / "yolo11n-pose.pt",
        Path.cwd() / "yolo11n-pose.pt",
        Path.cwd() / "hockeypose_1" / "yolo11n-pose.pt",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return "yolo11n-pose.pt"


def parse_source(src: str):
    # If it's an existing path, return as-is. If it's an int (string), treat as webcam index.
    if os.path.exists(src):
        return src
    try:
        return int(src)
    except ValueError:
        return src
def _people_count_from_result(res) -> int:
    # Prefer keypoints count if available; otherwise fallback to boxes count.
    kp = getattr(res, "keypoints", None)
    if kp is not None:
        xy = getattr(kp, "xy", None)
        if xy is not None:
            try:
                return len(xy)
            except Exception:
                pass
    boxes = getattr(res, "boxes", None)
    try:
        return len(boxes) if boxes is not None else 0
    except Exception:
        return 0


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Realtime YOLOv11 Pose with metrics (webcam/video)")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g., 0) or path to video file")
    parser.add_argument("--model", type=str, default=guess_default_model(script_dir), help="Path to yolo11n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="", help="Device for inference, e.g. 'cpu', 'cuda', '0'")
    parser.add_argument("--nosync", action="store_true", help="Do not sync to source FPS; run as fast as possible")
    parser.add_argument("--width", type=int, default=0, help="Optional resize width for processing/display (keep aspect)")
    parser.add_argument("--no-draw", action="store_true", help="Disable model's annotation drawing for speed")
    parser.add_argument("--testframe", action="store_true", help="Run one-frame smoketest headless and exit")
    args = parser.parse_args()

    cap_src = parse_source(args.source)
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open source: {args.source}")
        return 1

    # Get nominal FPS (for RT-factor and optional sync); for webcam often 0 -> fallback
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if src_fps <= 0 or src_fps > 240:
        src_fps = 30.0
    target_dt = 1.0 / src_fps

    # Prepare model
    model_path = args.model
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    model = YOLO(model_path)

    # Metrics accumulators
    last_n = 60
    inf_ms_hist = deque(maxlen=last_n)
    loop_dt_hist = deque(maxlen=last_n)

    frame_idx = 0
    paused = False
    sync_to_fps = not args.nosync

    window_title = "YOLOv11 Pose - Realtime metrics (q: quit, p: pause, s: sync)"

    # Optional headless smoketest: read one frame, run inference, print result, exit
    if args.testframe:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Could not read a frame from source for test.")
            return 2
        if args.width and frame.shape[1] != args.width:
            new_w = args.width
            new_h = int(frame.shape[0] * (new_w / frame.shape[1]))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device or None, verbose=False)[0]
        t1 = time.perf_counter()
        people = _people_count_from_result(res)
        print(f"OK testframe: people={people}, inf_ms={(t1 - t0)*1000.0:.2f}")
        return 0

    while True:
        frame_start = time.perf_counter()

        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            # Optional resize for speed
            if args.width and frame.shape[1] != args.width:
                new_w = args.width
                new_h = int(frame.shape[0] * (new_w / frame.shape[1]))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Inference
            t0 = time.perf_counter()
            res = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device or None, verbose=False)[0]
            t1 = time.perf_counter()
            inf_ms = (t1 - t0) * 1000.0
            inf_ms_hist.append(inf_ms)

            # Draw model annotations
            if not args.no_draw:
                # Ultralytics returns an annotated image via .plot()
                frame = res.plot()

            # People count
            people = _people_count_from_result(res)
        else:
            inf_ms = 0.0
            people = 0

        # Metrics
        loop_dt = time.perf_counter() - frame_start
        loop_dt_hist.append(loop_dt)

        avg_inf_ms = sum(inf_ms_hist) / len(inf_ms_hist) if inf_ms_hist else 0.0
        avg_fps = (1.0 / (sum(loop_dt_hist) / len(loop_dt_hist))) if loop_dt_hist else 0.0
        rt_factor = loop_dt / target_dt if target_dt > 0 else 0.0

        # Overlay metrics
        y = 24
        for line in [
            f"Frame: {frame_idx}",
            f"Source FPS: {src_fps:.1f}",
            f"Sync to FPS: {'ON' if sync_to_fps else 'OFF'}",
            f"Inference: {format_ms(inf_ms)} (avg {format_ms(avg_inf_ms)})",
            f"Pipeline FPS: {avg_fps:.1f}",
            f"RT factor: {rt_factor:.2f}x (<=1 is realtime)",
            f"Res: {frame.shape[1]}x{frame.shape[0]}",
            f"People: {people}",
        ]:
            draw_text(frame, line, 10, y)
            y += 22

        cv2.imshow(window_title, frame)

        if sync_to_fps and not paused:
            remain = target_dt - (time.perf_counter() - frame_start)
            if remain > 0:
                time.sleep(remain * 0.95)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            sync_to_fps = not sync_to_fps

        if not paused:
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
