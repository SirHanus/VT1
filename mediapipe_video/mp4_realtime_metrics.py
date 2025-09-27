import argparse
import time
import os
from collections import deque

import cv2
import mediapipe as mp


# Simple real-time-ish MediaPipe Pose runner for MP4 files with on-screen metrics.
# Usage (Windows cmd):
#   py -3 d:\WORK\VT1\webcam_tests\mediapipe_video\mp4_realtime_metrics.py --video d:\WORK\VT1\hockeypose_1\data_hockey.mp4
# Optional args:
#   --nosync   Disable syncing playback to the video's FPS (run as fast as possible)
#   --width    Resize width for processing/display (keeps aspect ratio)
#   --no-draw  Disable landmark drawing for speed

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles


def format_ms(ms: float) -> str:
    return f"{ms:.1f} ms"


def draw_text(img, text, x, y, scale=0.6, color=(255, 255, 255), thickness=1):
    # Draw black outline then white text for readability
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Run MediaPipe Pose on an MP4 with realtime-like playback and metrics.")
    parser.add_argument("--video", type=str, default=os.path.join("/", "hockeypose_1", "data_hockey.mp4"),
                        help="Path to input MP4 video")
    parser.add_argument("--nosync", action="store_true", help="Do not sync to source FPS; run as fast as possible")
    parser.add_argument("--width", type=int, default=0, help="Optional resize width for processing/display (keep aspect)")
    parser.add_argument("--no-draw", action="store_true", help="Disable landmark drawing for speed")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return 1

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    target_dt = 1.0 / float(src_fps) if src_fps > 0 else 1.0 / 30.0

    # Rolling windows for metrics
    last_n = 60
    inf_ms_hist = deque(maxlen=last_n)
    loop_dt_hist = deque(maxlen=last_n)

    # Pose configuration mirrors webcam script defaults

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        paused = False
        sync_to_fps = not args.nosync

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

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                t0 = time.perf_counter()
                result = pose.process(rgb)
                t1 = time.perf_counter()
                inf_ms = (t1 - t0) * 1000.0
                inf_ms_hist.append(inf_ms)

                if (result.pose_landmarks is not None) and (not args.no_draw):
                    mp_draw.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_style.get_default_pose_landmarks_style(),
                    )

                frame_idx += 1
            else:
                # When paused, we still show last frame and allow resume
                inf_ms = 0.0

            # Metrics and overlay
            now = time.perf_counter()
            loop_dt = now - frame_start
            loop_dt_hist.append(loop_dt)

            avg_inf_ms = sum(inf_ms_hist) / len(inf_ms_hist) if inf_ms_hist else 0.0
            avg_fps = (1.0 / (sum(loop_dt_hist) / len(loop_dt_hist))) if loop_dt_hist else 0.0
            rt_factor = loop_dt / target_dt if target_dt > 0 else 0.0

            # Draw metrics box
            metrics = [
                f"Frame: {frame_idx}",
                f"Source FPS: {src_fps:.1f}",
                f"Sync to FPS: {'ON' if sync_to_fps else 'OFF'}",
                f"Inference: {format_ms(inf_ms)} (avg {format_ms(avg_inf_ms)})",
                f"Pipeline FPS: {avg_fps:.1f}",
                f"RT factor: {rt_factor:.2f}x (<=1 is realtime)",
                f"Res: {frame.shape[1]}x{frame.shape[0]}",
            ]

            y = 24
            for line in metrics:
                draw_text(frame, line, 10, y)
                y += 22

            cv2.imshow("MediaPipe Pose - MP4 realtime metrics (q: quit, p: pause, s: sync toggle)", frame)

            # Realtime sync: sleep to match source FPS if processing is faster than target
            if sync_to_fps and not paused:
                elapsed = time.perf_counter() - frame_start
                remain = target_dt - elapsed
                if remain > 0:
                    # Sleep just a bit less than remain to avoid oversleeping
                    time.sleep(remain * 0.95)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('s'):
                sync_to_fps = not sync_to_fps

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

