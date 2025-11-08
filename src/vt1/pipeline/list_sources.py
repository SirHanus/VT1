# python
import argparse
import os
import sys
from pathlib import Path
from threading import Thread

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}


def try_read_with_timeout(cap, timeout=1.0):
    res = {"ok": False, "frame": None}

    def worker():
        try:
            ok, frame = cap.read()
            res["ok"], res["frame"] = ok, frame
        except Exception:
            res["ok"], res["frame"] = False, None

    t = Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    return res["ok"], res["frame"]


def probe_camera_index(i: int, timeout=1.0):
    backends = []
    # Prefer Windows backends
    if os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    else:
        backends = [0]  # default
    for be in backends:
        cap = cv2.VideoCapture(i, be) if be != 0 else cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = try_read_with_timeout(cap, timeout=timeout)
        if ok and frame is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()
            return {
                "index": i,
                "backend": (
                    "CAP_DSHOW"
                    if be == cv2.CAP_DSHOW
                    else ("CAP_MSMF" if be == cv2.CAP_MSMF else "DEFAULT")
                ),
                "width": w,
                "height": h,
                "fps": float(fps),
            }
        cap.release()
    return None


def probe_cameras(max_index=10, timeout=1.0):
    found = []
    for i in range(max_index + 1):
        info = probe_camera_index(i, timeout=timeout)
        if info:
            found.append(info)
    return found


def is_video_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


def probe_video_file(p: Path, timeout=1.0):
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = try_read_with_timeout(cap, timeout=timeout)
    if not ok or frame is None:
        cap.release()
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        else 0
    )
    duration = (count / fps) if fps > 0 and count > 0 else 0.0
    cap.release()
    return {
        "path": str(p),
        "width": w,
        "height": h,
        "fps": float(fps),
        "frames": count,
        "duration_sec": duration,
    }


def scan_video_files(root: Path, recursive=True, timeout=1.0):
    files = []
    it = root.rglob("*") if recursive else root.glob("*")
    for p in it:
        if is_video_file(p):
            info = probe_video_file(p, timeout=timeout)
            if info:
                files.append(info)
    return files


def get_windows_camera_names():
    if os.name != "nt":
        return []
    try:
        import subprocess, json

        # Query PnP cameras via PowerShell
        ps = r"Get-CimInstance Win32_PnPEntity | Where-Object {$_.PNPClass -in @('Camera','Image')} | Select-Object -ExpandProperty Name | ConvertTo-Json"
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        data = json.loads(out.decode("utf-8"))
        if isinstance(data, list):
            return data
        elif isinstance(data, str):
            return [data]
    except Exception:
        pass
    return []


def main():
    ap = argparse.ArgumentParser("List usable OpenCV sources for --source")
    ap.add_argument(
        "--max-index", type=int, default=10, help="Max webcam index to probe"
    )
    ap.add_argument(
        "--scan-dir", type=str, default=".", help="Directory to scan for video files"
    )
    ap.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive scan for video files",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Seconds to wait when grabbing a frame",
    )
    args = ap.parse_args()

    print("Probing cameras...")
    cams = probe_cameras(max_index=args.max_index, timeout=args.timeout)
    if cams:
        for c in cams:
            print(
                f"- Camera index {c['index']} [{c['backend']}], {c['width']}x{c['height']}, fps~{c['fps']:.1f}  -> use --source {c['index']}"
            )
    else:
        print("- No cameras found")

    if os.name == "nt":
        names = get_windows_camera_names()
        if names:
            print("Detected Windows camera devices:")
            for n in names:
                print(f"  - {n}")

    root = Path(args.scan_dir).resolve()
    print(
        f"Scanning video files under {root} ({'non-recursive' if args.no_recursive else 'recursive'})..."
    )
    vids = scan_video_files(root, recursive=not args.no_recursive, timeout=args.timeout)
    if vids:
        for v in vids:
            dur = f", {v['duration_sec']:.1f}s" if v["duration_sec"] > 0 else ""
            print(
                f"- File {v['path']}  ({v['width']}x{v['height']}, fps~{v['fps']:.1f}{dur})  -> use --source {v['path']}"
            )
    else:
        print("- No readable video files found")

    print("\nExamples you can pass to --source:")
    print("- Integer webcam index, e.g. 0")
    print("- Path to a video file, e.g. C:\\path\\to\\video.mp4")
    print(
        "- Network streams supported by OpenCV/FFmpeg, e.g. rtsp://user:pass@host:554/stream (if available)"
    )


if __name__ == "__main__":
    sys.exit(main())
