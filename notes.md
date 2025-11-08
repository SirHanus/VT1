# VT1 - Hockey Video Analysis Pipeline

## Development Timeline

### October 3, 2024
- Base demos with Camera input for:
    - MediaPipe Pose
    - OpenVINO MoveNet
    - TRT Pose
    - PyTorch Keypoint Baseline
    - MMPose

### October 16, 2024
- Minimal YOLO Pose + SAM2 POC on data_hockey.mp4 (offline and online)
- Pass YOLO poses into SAM2 for mask generation
- Performance and memory reset options added (run larger videos reliably, try different model sizes)
- Batch run script
- Logging and progress bars (saving in JSON)

### October 30, 2024
- Add unsupervised clustering (into teams) - separate pipeline
- ~~Maybe add number identification? connect with skeleton~~
- ~~Try to run it on bigger GPU and parallelize?~~

### October 31, 2024
- Team clustering integrated - Full workflow from data generation, training to inference
- Per-match model training - Build custom clustering models per game
- Evaluation tools - Visual validation of team separation quality


### November 8, 2024 (Current)
- [x] Team clustering fully integrated
- [x] PyQt6 GUI with startup workflow automation
- [x] Configuration system with TOML and environment variables

### November 14, 2024
- [ ] Starting writing the paper (some chapter split etc.)
- [ ] Finetune YOLO model? Make dataset and improve it (YOLO performance currently affects other parts too)





