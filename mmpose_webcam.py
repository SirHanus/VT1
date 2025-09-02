import cv2
from mmpose.apis import MMPoseInferencer
import numpy as np

# rtmpose-s + rtmdet-tiny are good speed/accuracy tradeoffs
inferencer = MMPoseInferencer(pose2d='human', device='cpu')  # Changed from 'cuda' to 'cpu'

# return_vis=True gives you RGB frames with skeletons drawn
for out in inferencer(inputs='webcam', return_vis=True, show=False):
    frame = out.get('visualization')
    if frame is None:  # warmup
        continue
    
    # Handle potential list structure by ensuring frame is a numpy array
    if isinstance(frame, list) and len(frame) > 0:
        frame = frame[0]  # Take the first frame if it's a list of frames
    
    # Make sure we have a valid numpy array before trying to reverse channels
    if isinstance(frame, np.ndarray):
        # MMPose returns RGB; OpenCV expects BGR
        bgr = frame[:, :, ::-1]
        cv2.imshow("MMPose (press 'q' to quit)", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print(f"Unexpected frame type: {type(frame)}")

cv2.destroyAllWindows()