#detecting hand gestures 

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mediapipe as mp
import cv2

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_HANDS = 1  

@dataclass
class Landmark:
    x: float
    y: float

class HandDetector:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=MAX_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, frame: cv2.Mat) -> Sequence[Landmark] | None:
        #convert the BGR image to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if results.multi_hand_landmarks is None:
            return None

        h, w, *_ = frame.shape
        hand = results.multi_hand_landmarks[0]
        return [Landmark(lm.x * w, lm.y * h) for lm in hand.landmark]

        