import cv2
from mmpose.apis import MMPoseInferencer
import numpy as np

# rtmpose-s + rtmdet-tiny are good speed/accuracy tradeoffs
inferencer = MMPoseInferencer(pose2d='human', device='cpu')  # Changed from 'cuda' to 'cpu'

# Set display=False to explicitly prevent MMPoseInferencer from showing its own window
for out in inferencer(inputs='webcam', return_vis=True, show=False):
    pass

