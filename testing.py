import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

import mmcv, mmdet, mmpose, mmengine
print("mmcv", mmcv.__version__)
print("mmdet", mmdet.__version__)
print("mmpose", mmpose.__version__)
print("mmengine", mmengine.__version__)

