# ML Backend for Label Studio

This directory contains a YOLO-based ML backend for Label Studio that enables automatic pre-labeling of images.

## Features

- **Detection**: Bounding box detection for objects
- **Pose Estimation**: Keypoint detection for human poses
- **Auto Pre-labeling**: Automatically generates predictions when importing images
- **Configurable**: Adjust model and confidence threshold via environment variables

## Configuration

Environment variables in `compose.yml`:

- `YOLO_MODEL`: Path to YOLO model (default: `yolo11n-pose.pt`)
- `CONF_THRESHOLD`: Confidence threshold for predictions (default: `0.25`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

## Usage

1. Start the services:
   ```bash
   docker compose up -d --build
   ```

2. The ML backend will start automatically at http://localhost:9090

3. In Label Studio (http://localhost:9001):
   - Login with username: `admin`, password: `admin`
   - Go to your project
   - Go to **Settings** → **Machine Learning**
   - Click **Add Model**
   - Enter URL: `http://yolo-backend:9090`
   - Click **Validate and Save**
   - Enable **Start model training on annotation submission** (optional)
   - Enable **Retrieve predictions when loading a task automatically**

4. Import images and the predictions will be automatically generated

## Troubleshooting

### Check backend status
```bash
docker logs vt1-yolo-backend
```

### Test backend health
```bash
curl http://localhost:9090/health
```

### Rebuild after changes
```bash
docker compose down
docker compose up -d --build
```

## Custom Models

To use your own YOLO model:

1. Place your model in the `models/` directory
2. Update `YOLO_MODEL` in `compose.yml` to point to `/models/your-model.pt`
3. Rebuild and restart: `docker compose up -d --build yolo-backend`

## Supported Label Studio Templates

### For Detection Models
```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="red"/>
    <Label value="ball" background="blue"/>
  </RectangleLabels>
</View>
```

### For Pose Models
```xml
<View>
  <Image name="image" value="$image"/>
  <KeyPointLabels name="keypoints" toName="image">
    <Label value="nose" background="red"/>
    <Label value="left_eye" background="blue"/>
    <Label value="right_eye" background="blue"/>
    <!-- Add all keypoint labels -->
  </KeyPointLabels>
</View>
```

## Troubleshooting

Check logs:
```bash
docker compose logs yolo-backend
```

Restart backend:
```bash
docker compose restart yolo-backend
```

