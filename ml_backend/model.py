import logging
import os

from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOBackend(LabelStudioMLBase):
    """YOLO ML Backend for Label Studio - supports detection and pose estimation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = os.environ.get("YOLO_MODEL", "yolo11n-pose.pt")
        self.model = YOLO(model_path)
        self.conf_threshold = float(os.environ.get("CONF_THRESHOLD", "0.25"))
        self.model_type = "pose" if "pose" in model_path else "detect"
        logger.info(
            f"Loaded YOLO model: {model_path} (type={self.model_type}, conf={self.conf_threshold})"
        )

    def predict(self, tasks, **kwargs):
        """Generate predictions for Label Studio tasks"""
        predictions = []

        for task in tasks:
            image_url = task["data"].get("image")
            if not image_url:
                predictions.append({"result": []})
                continue

            # Handle local file paths from Label Studio
            if image_url.startswith("/data/local-files/?d="):
                # Extract the relative path from Label Studio URL
                import urllib.parse

                parsed = urllib.parse.urlparse(image_url)
                params = urllib.parse.parse_qs(parsed.query)
                rel_path = params.get("d", [""])[0]
                image_path = os.path.join("/label-studio/data", rel_path)
            elif image_url.startswith("/data/"):
                image_path = image_url.replace("/data/", "/label-studio/data/")
            else:
                image_path = image_url

            try:
                results = self.model(image_path, conf=self.conf_threshold)[0]
                result = []
                img_width, img_height = results.orig_shape[1], results.orig_shape[0]

                # Handle detection boxes
                if (
                    hasattr(results, "boxes")
                    and results.boxes is not None
                    and len(results.boxes) > 0
                ):
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = self.model.names[cls_id]

                        result.append(
                            {
                                "from_name": "label",
                                "to_name": "image",
                                "type": "rectanglelabels",
                                "value": {
                                    "rectanglelabels": [label],
                                    "x": (x1 / img_width) * 100,
                                    "y": (y1 / img_height) * 100,
                                    "width": ((x2 - x1) / img_width) * 100,
                                    "height": ((y2 - y1) / img_height) * 100,
                                },
                                "score": conf,
                            }
                        )

                # Handle pose keypoints
                if (
                    self.model_type == "pose"
                    and hasattr(results, "keypoints")
                    and results.keypoints is not None
                ):
                    keypoint_names = [
                        "nose",
                        "left_eye",
                        "right_eye",
                        "left_ear",
                        "right_ear",
                        "left_shoulder",
                        "right_shoulder",
                        "left_elbow",
                        "right_elbow",
                        "left_wrist",
                        "right_wrist",
                        "left_hip",
                        "right_hip",
                        "left_knee",
                        "right_knee",
                        "left_ankle",
                        "right_ankle",
                    ]

                    for person_idx, kpts in enumerate(results.keypoints.xy):
                        for kpt_idx, (x, y) in enumerate(kpts):
                            if x > 0 and y > 0:  # Valid keypoint
                                kpt_name = (
                                    keypoint_names[kpt_idx]
                                    if kpt_idx < len(keypoint_names)
                                    else f"keypoint_{kpt_idx}"
                                )
                                result.append(
                                    {
                                        "from_name": "keypoints",
                                        "to_name": "image",
                                        "type": "keypointlabels",
                                        "value": {
                                            "x": (float(x) / img_width) * 100,
                                            "y": (float(y) / img_height) * 100,
                                            "keypointlabels": [kpt_name],
                                            "width": 1.0,
                                        },
                                        "score": (
                                            results.boxes[person_idx].conf[0].item()
                                            if person_idx < len(results.boxes)
                                            else 0.9
                                        ),
                                    }
                                )

                max_conf = max([r.get("score", 0) for r in result], default=0)
                predictions.append({"result": result, "score": max_conf})
                logger.info(f"Processed {image_path}: {len(result)} annotations")

            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
                predictions.append({"result": []})

        return predictions

    def fit(self, event, data, **kwargs):
        """Optional: implement fine-tuning based on Label Studio annotations"""
        logger.info(f"Fit called with event={event}, but training not implemented yet")
        pass
