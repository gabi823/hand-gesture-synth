# main file for the application

from __future__ import annotations

import sys
import cv2
import mediapipe as mp

CAM_INDEX = 0
WIDTH = 640
HEIGHT = 480
FLIP = True

from hand_detector import HandDetector
from gesture_classifier import classify
from synth import Synth

#draw overlays on the frame??


#main loop
def main() -> None:

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    detector = HandDetector()
    synth = Synth()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if FLIP:
            frame = cv2.flip(frame, 1)

        # detect and classify hand gestures
        landmarks = detector.detect(frame)
        gesture = classify(landmarks) if landmarks else None

        #sound
        synth.play(gesture)

        #draw
        #_draw_overlays(frame, landmarks, gesture)

        cv2.imshow("Hand Gesture Synth", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27): # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("bye bye")



if __name__ == "__main__":
    main()