import cv2, torch
import torchvision
import torchvision.transforms.functional as F

# COCO keypoint skeleton (17 joints)
SKELETON = [
    (15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),
    (5,6),(5,7),(7,9),(6,8),(8,10),(1,2),(0,1),(0,2),(1,3),(2,4),(3,5),(4,6)
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights).to(device).eval()

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    img = F.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
    with torch.no_grad():
        out = model([img])[0]

    boxes, scores = out['boxes'].cpu(), out['scores'].cpu()
    keypoints = out['keypoints'].cpu()  # [N, 17, 3] (x,y,score)
    for i in range(len(boxes)):
        if scores[i] < 0.7:
            continue
        kps = keypoints[i]
        # draw joints
        for j in range(kps.shape[0]):
            x, y, s = kps[j]
            if s > 0.5:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        # draw limbs
        for a, b in SKELETON:
            xa, ya, sa = kps[a]
            xb, yb, sb = kps[b]
            if sa > 0.5 and sb > 0.5:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)

    cv2.imshow("TorchVision Keypoint R-CNN (press 'q' to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
