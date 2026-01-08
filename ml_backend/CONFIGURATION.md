# YOLO ML Backend Configuration Guide

## Model Configuration

### Using Different YOLO Models

Edit `compose.yml` and change the `YOLO_MODEL` environment variable:

#### Built-in Models (automatically downloaded)
```yaml
environment:
  - YOLO_MODEL=yolo11n-pose.pt    # Nano pose model (fastest)
  - YOLO_MODEL=yolo11s-pose.pt    # Small pose model
  - YOLO_MODEL=yolo11m-pose.pt    # Medium pose model
  - YOLO_MODEL=yolo11l-pose.pt    # Large pose model
  - YOLO_MODEL=yolo11x-pose.pt    # Extra large pose model (best accuracy)
  
  # Detection models (bounding boxes only)
  - YOLO_MODEL=yolo11n.pt         # Nano detection
  - YOLO_MODEL=yolo11s.pt         # Small detection
  - YOLO_MODEL=yolo11m.pt         # Medium detection
  - YOLO_MODEL=yolo11l.pt         # Large detection
  - YOLO_MODEL=yolo11x.pt         # Extra large detection
```

#### Custom/Fine-tuned Models
Place your model in the `models/` directory:
```yaml
volumes:
  - ./models:/models:ro

environment:
  - YOLO_MODEL=/models/my-finetuned-model.pt
```

### Confidence Threshold

Adjust the confidence threshold for predictions:
```yaml
environment:
  - CONF_THRESHOLD=0.25    # Default: 0.25 (25%)
  - CONF_THRESHOLD=0.50    # Stricter: only high-confidence predictions
  - CONF_THRESHOLD=0.10    # More predictions: include lower confidence
```

## Label Studio Configuration

### For Pose Estimation Projects

Use this labeling configuration:
```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="red"/>
  </RectangleLabels>
  <KeyPointLabels name="keypoints" toName="image">
    <Label value="nose" background="#FF0000"/>
    <Label value="left_eye" background="#00FF00"/>
    <Label value="right_eye" background="#00FF00"/>
    <Label value="left_ear" background="#0000FF"/>
    <Label value="right_ear" background="#0000FF"/>
    <Label value="left_shoulder" background="#FFFF00"/>
    <Label value="right_shoulder" background="#FFFF00"/>
    <Label value="left_elbow" background="#FF00FF"/>
    <Label value="right_elbow" background="#FF00FF"/>
    <Label value="left_wrist" background="#00FFFF"/>
    <Label value="right_wrist" background="#00FFFF"/>
    <Label value="left_hip" background="#FFA500"/>
    <Label value="right_hip" background="#FFA500"/>
    <Label value="left_knee" background="#800080"/>
    <Label value="right_knee" background="#800080"/>
    <Label value="left_ankle" background="#008080"/>
    <Label value="right_ankle" background="#008080"/>
  </KeyPointLabels>
</View>
```

### For Object Detection Projects

Use this labeling configuration:
```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="red"/>
    <Label value="bicycle" background="blue"/>
    <Label value="car" background="green"/>
    <Label value="motorcycle" background="yellow"/>
    <Label value="ball" background="orange"/>
    <!-- Add more COCO classes as needed -->
  </RectangleLabels>
</View>
```

## Workflow

### 1. Import Images
- Go to your Label Studio project
- Click **Import** → **Upload Files**
- Select images from your local machine or use local file serving

### 2. Get Predictions
- Click on a task to open the labeling interface
- Click **Get Predictions** button (or enable auto-prediction)
- The YOLO model will generate pre-labels

### 3. Review and Correct
- Review the auto-generated predictions
- Correct any mistakes
- Submit the annotation

### 4. Export for Training
- Go to **Export** in project settings
- Choose format (YOLO, COCO, etc.)
- Use exported data to fine-tune your model

## Performance Tips

1. **Start with a small model** (nano/small) for faster initial labeling
2. **Adjust confidence threshold** based on your needs:
   - Lower (0.1-0.2): More predictions, more false positives
   - Higher (0.5-0.7): Fewer predictions, higher precision
3. **Use GPU** for faster inference (automatically used if available)
4. **Fine-tune on your data** after labeling a few hundred samples

## Troubleshooting

### Backend not connecting
```powershell
# Check if container is running
docker ps | Select-String yolo-backend

# View logs
docker compose logs yolo-backend

# Restart backend
docker compose restart yolo-backend
```

### Wrong predictions
- Lower the confidence threshold in compose.yml
- Use a larger model (e.g., switch from 'n' to 's' or 'm')
- Fine-tune the model on your specific data

### Out of memory errors
- Use a smaller model (nano or small)
- Reduce image size in Label Studio settings
- Allocate more memory to Docker Desktop

## Advanced: Using Fine-tuned Models

After training your model with the VT1 pipeline:

1. Copy your fine-tuned model:
   ```powershell
   Copy-Item "runs/pose/train/weights/best.pt" -Destination "models/hockey-pose-finetuned.pt"
   ```

2. Update compose.yml:
   ```yaml
   environment:
     - YOLO_MODEL=/models/hockey-pose-finetuned.pt
   ```

3. Rebuild and restart:
   ```powershell
   docker compose up -d --build yolo-backend
   ```

