import sys
try:
    import mediapipe
    print("MediaPipe version:", mediapipe.__version__)
    import mediapipe.solutions
    print("Has solutions.pose:", hasattr(mediapipe.solutions, "pose"))
    if hasattr(mediapipe.solutions, "pose"):
        print("SUCCESS: MediaPipe is properly installed")
    else:
        print("ERROR: mediapipe.solutions.pose not found")
except Exception as e:
    print("ERROR:", e)
