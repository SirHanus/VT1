# openvino_pose_webcam.py
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from openvino.runtime import Core

# Model expects BGR input 1x3x256x456 and outputs:
#  - PAFs:   [1, 38, 32, 57]  (unused here)
#  - Heatmaps [1, 19, 32, 57] (18 body parts + background)
# Ref: OpenVINO OMZ model docs.  https://docs.openvino.ai/...  (see citation in chat)

# OpenPose/COCO 18-keypoint order used by human-pose-estimation-0001
KPT_NAMES = [
    "Nose","Neck",
    "RShoulder","RElbow","RWrist",
    "LShoulder","LElbow","LWrist",
    "RHip","RKnee","RAnkle",
    "LHip","LKnee","LAnkle",
    "REye","LEye","REar","LEar"
]

# Skeleton edges (pairs of indices) to draw
POSE_PAIRS = [
    (1,2),(1,5),
    (2,3),(3,4),(5,6),(6,7),
    (1,8),(8,9),(9,10),
    (1,11),(11,12),(12,13),
    (1,0),(0,14),(14,16),(0,15),(15,17)
]

def parse_args():
    ap = argparse.ArgumentParser(description="OpenVINO Pose Detection (webcam)")
    ap.add_argument("--model", "-m", type=str, default="intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml",
                    help="Path to model .xml (IR) file from Open Model Zoo")
    ap.add_argument("--device", "-d", type=str, default="AUTO", help="Device: AUTO|CPU|GPU|NPU etc.")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index")
    ap.add_argument("--thr", type=float, default=0.15, help="Heatmap confidence threshold")
    ap.add_argument("--display-size", type=str, default="912x512",
                    help="Display size WxH for the window (resized view)")
    return ap.parse_args()

def letterbox_resize(img, dst_w=456, dst_h=256):
    """Resize with padding to keep aspect ratio; returns resized image and metadata to map back."""
    h, w = img.shape[:2]
    scale = min(dst_w / w, dst_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((dst_h, dst_w, 3), dtype=img.dtype)
    top = (dst_h - nh) // 2
    left = (dst_w - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    meta = dict(scale=scale, top=top, left=left, in_w=dst_w, in_h=dst_h, orig_w=w, orig_h=h)
    return canvas, meta

def heatmaps_argmax_decode(heatmaps, conf_thr=0.15):
    """
    Simple single-peak decode per keypoint channel (ignores PAFs).
    heatmaps: [19, H, W] (np.float32). Channel 18 is background -> ignore.
    Returns list of (x, y, score) for 18 keypoints in heatmap coordinates.
    """
    kpts = []
    part_maps = heatmaps[:-1]  # drop background
    _, H, W = part_maps.shape
    for c in range(part_maps.shape[0]):
        hm = part_maps[c]
        idx = np.argmax(hm)
        y, x = np.unravel_index(idx, hm.shape)
        score = hm[y, x]
        if score < conf_thr:
            kpts.append((None, None, 0.0))
        else:
            kpts.append((float(x), float(y), float(score)))
    return kpts, (W, H)

def map_to_input_space(kpts, hm_size, input_size):
    """Map heatmap coords to model input (456x256) coords."""
    W_hm, H_hm = hm_size
    in_w, in_h = input_size
    scale_x = in_w / W_hm
    scale_y = in_h / H_hm
    mapped = []
    for x, y, s in kpts:
        if x is None:
            mapped.append((None, None, 0.0))
        else:
            mapped.append((x * scale_x, y * scale_y, s))
    return mapped

def map_to_frame_space(kpts_in, meta):
    """Map model-input coords back to original frame coords (invert letterbox)."""
    out = []
    for x, y, s in kpts_in:
        if x is None:
            out.append((None, None, 0.0))
        else:
            x0 = (x - meta["left"]) / meta["scale"]
            y0 = (y - meta["top"]) / meta["scale"]
            out.append((x0, y0, s))
    return out

def draw_pose(frame, kpts, pairs=POSE_PAIRS, point_thr=0.15):
    """Draw keypoints and skeleton on frame (in original frame coords)."""
    for i, (x, y, s) in enumerate(kpts):
        if x is None or s < point_thr:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
        cv2.putText(frame, str(i), (int(x)+4, int(y)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    for a, b in pairs:
        xa, ya, sa = kpts[a]
        xb, yb, sb = kpts[b]
        if None in (xa, ya, xb, yb):
            continue
        if sa < point_thr or sb < point_thr:
            continue
        cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)

# Helper to robustly get dimension sizes from a port (works for static or dynamic shapes)
def _dims_from_port(port):
    """
    Returns a list of ints for dimensions.
    - For static shapes, returns concrete ints (e.g., [1, 19, 32, 57]).
    - For dynamic dimensions, returns -1 in that position.
    """
    # Try straightforward static shape first
    try:
        shape = port.shape  # often a tuple/list of ints
        if isinstance(shape, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in shape):
            return list(map(int, shape))
    except Exception:
        pass

    # Fallback to partial_shape if available
    ps = getattr(port, "partial_shape", None)
    if ps is not None:
        dims = []
        for d in ps:
            # Dimension API: d.is_dynamic, d.get_length()
            try:
                if getattr(d, "is_dynamic", False):
                    dims.append(-1)
                else:
                    dims.append(int(d.get_length()))
            except Exception:
                # As a last resort, try direct int()
                try:
                    dims.append(int(d))
                except Exception:
                    dims.append(-1)
        return dims

    # Last resort: treat unknown as dynamic
    return [-1, -1, -1, -1]

def pick_heatmaps_output(outputs):
    """
    Model has two outputs: PAFs [1,38,H,W] and heatmaps [1,19,H,W].
    Return the heatmaps output blob (index).
    """
    idx = None
    for i, out in enumerate(outputs):
        dims = _dims_from_port(out)
        if len(dims) == 4 and dims[1] == 19:
            idx = i
            break
    if idx is None:
        # Fallback: choose the one with fewer channels (19 vs 38)
        sizes = []
        for i, out in enumerate(outputs):
            dims = _dims_from_port(out)
            c = dims[1] if len(dims) >= 2 else 9999
            c = c if c >= 0 else 9999  # treat dynamic as large number
            sizes.append((i, c))
        idx = min(sizes, key=lambda t: t[1])[0]
    return idx

def main():
    args = parse_args()
    disp_w, disp_h = map(int, args.display_size.lower().split("x"))

    model_xml = Path(args.model)
    if not model_xml.exists():
        raise FileNotFoundError(
            f"Model not found at {model_xml}\n"
            "Use OMZ tools to get it:\n"
            "  omz_downloader --name human-pose-estimation-0001\n"
            "  omz_converter  --name human-pose-estimation-0001"
        )

    ie = Core()
    compiled = ie.compile_model(model=model_xml.as_posix(), device_name=args.device)
    input_port = compiled.inputs[0]
    outputs = compiled.outputs
    heatmaps_out_idx = pick_heatmaps_output(outputs)

    # Robustly parse input dims
    in_n, in_c, in_h, in_w = _dims_from_port(input_port)
    if (in_h, in_w) != (256, 456):
        # Some variants may have different resolutions; warn but continue.
        # You can remove this check if running with non-standard input sizes.
        raise AssertionError(f"Unexpected model input size: {(in_h, in_w)}. Expected (256, 456).")

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows; harmless elsewhere
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.cam}")

    win_name = "OpenVINO Pose (human-pose-estimation-0001)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    t_last = time.time()
    frames = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Preprocess to model input
            pre, meta = letterbox_resize(frame, dst_w=in_w, dst_h=in_h)
            blob = pre.transpose(2, 0, 1)[None].astype(np.float32)  # 1x3xHxW

            # Inference
            res = compiled([blob])
            heatmaps = res[outputs[heatmaps_out_idx]].squeeze(0)  # [19, 32, 57]

            # Decode (simple argmax per part)
            kpts_hm, hm_size = heatmaps_argmax_decode(heatmaps, conf_thr=args.thr)
            kpts_in = map_to_input_space(kpts_hm, hm_size, (in_w, in_h))
            kpts_fr = map_to_frame_space(kpts_in, meta)

            # Draw on a resized display frame to keep aspect clean
            disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            # Scale keypoints to display size
            sx, sy = disp_w / frame.shape[1], disp_h / frame.shape[0]
            kpts_disp = []
            for x, y, s in kpts_fr:
                if x is None:
                    kpts_disp.append((None, None, 0.0))
                else:
                    kpts_disp.append((x * sx, y * sy, s))

            draw_pose(disp, kpts_disp, point_thr=args.thr)

            # FPS
            frames += 1
            t_now = time.time()
            if t_now - t_last >= 0.5:
                fps = frames / (t_now - t_last)
                frames = 0
                t_last = t_now
            cv2.putText(disp, f"FPS: {fps:.1f}  Device: {args.device}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(win_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()