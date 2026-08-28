# hand gesture to music note logic

from __future__ import annotations

import math
from typing import Callable, Sequence

WRIST       = 0
THUMB_CMC   = 1
THUMB_MCP   = 2
THUMB_IP    = 3
THUMB_TIP   = 4
INDEX_MCP   = 5
INDEX_PIP   = 6
INDEX_DIP   = 7
INDEX_TIP   = 8
MIDDLE_MCP  = 9
MIDDLE_PIP  = 10
MIDDLE_DIP  = 11
MIDDLE_TIP  = 12
RING_MCP    = 13
RING_PIP    = 14
RING_DIP    = 15
RING_TIP    = 16
PINKY_MCP   = 17
PINKY_PIP   = 18
PINKY_DIP   = 19
PINKY_TIP   = 20

FINGER_EXTENSION_RATIO = 0.15


def _distance(a: Landmark, b: Landmark) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _finger_extended(
    tip: Landmark, pip: Landmark, wrist: Landmark, threshold: float,
) -> bool:
    """
    Return True when *tip* is sufficiently higher than *pip*.

    The required gap scales with the hand size (wrist → middle-finger PIP)
    so the same threshold works at different distances from the camera.
    """
    return (pip.y - tip.y) > threshold


def _thumb_extended(
    landmarks: Sequence[Landmark], wrist: Landmark, threshold: float,
) -> bool:
    """Thumb is extended when its tip is far from the wrist in x-direction."""
    # Compare the straight-line distance from wrist — extended thumb reaches
    # noticeably further.
    tip_dist = _distance(wrist, landmarks[THUMB_TIP])
    ip_dist = _distance(wrist, landmarks[THUMB_IP])
    return (tip_dist - ip_dist) > threshold * 0.8





def _is_open_palm(landmarks: Sequence[Landmark]) -> bool:
    wrist = landmarks[WRIST]
    palm_h = _distance(wrist, landmarks[MIDDLE_PIP])
    threshold = palm_h * FINGER_EXTENSION_RATIO

    fingers = [
        _finger_extended(landmarks[INDEX_TIP], landmarks[INDEX_PIP], wrist, threshold),
        _finger_extended(landmarks[MIDDLE_TIP], landmarks[MIDDLE_PIP], wrist, threshold),
        _finger_extended(landmarks[RING_TIP], landmarks[RING_PIP], wrist, threshold),
        _finger_extended(landmarks[PINKY_TIP], landmarks[PINKY_PIP], wrist, threshold),
        _thumb_extended(landmarks, wrist, threshold),
    ]
    return all(fingers)

def _is_fist(landmarks: Sequence[Landmark]) -> bool:
    wrist = landmarks[WRIST]
    palm_h = _distance(wrist, landmarks[MIDDLE_PIP])
    threshold = palm_h * FINGER_EXTENSION_RATIO

    fingers = [
        _finger_extended(landmarks[INDEX_TIP], landmarks[INDEX_PIP], wrist, threshold),
        _finger_extended(landmarks[MIDDLE_TIP], landmarks[MIDDLE_PIP], wrist, threshold),
        _finger_extended(landmarks[RING_TIP], landmarks[RING_PIP], wrist, threshold),
        _finger_extended(landmarks[PINKY_TIP], landmarks[PINKY_PIP], wrist, threshold),
        _thumb_extended(landmarks, wrist, threshold),
    ]
    return not any(fingers)

def _is_peace(landmarks: Sequence[Landmark]) -> bool:
    wrist = landmarks[WRIST]
    palm_h = _distance(wrist, landmarks[MIDDLE_PIP])
    threshold = palm_h * FINGER_EXTENSION_RATIO

    return (
        _finger_extended(landmarks[INDEX_TIP], landmarks[INDEX_PIP], wrist, threshold)
        and _finger_extended(landmarks[MIDDLE_TIP], landmarks[MIDDLE_PIP], wrist, threshold)
        and not _finger_extended(landmarks[RING_TIP], landmarks[RING_PIP], wrist, threshold)
        and not _finger_extended(landmarks[PINKY_TIP], landmarks[PINKY_PIP], wrist, threshold)
        and not _thumb_extended(landmarks, wrist, threshold)
    )


def _is_point(landmarks: Sequence[Landmark]) -> bool:
    wrist = landmarks[WRIST]
    palm_h = _distance(wrist, landmarks[MIDDLE_PIP])
    threshold = palm_h * FINGER_EXTENSION_RATIO

    return (
        _finger_extended(landmarks[INDEX_TIP], landmarks[INDEX_PIP], wrist, threshold)
        and not _finger_extended(landmarks[MIDDLE_TIP], landmarks[MIDDLE_PIP], wrist, threshold)
        and not _finger_extended(landmarks[RING_TIP], landmarks[RING_PIP], wrist, threshold)
        and not _finger_extended(landmarks[PINKY_TIP], landmarks[PINKY_PIP], wrist, threshold)
        and not _thumb_extended(landmarks, wrist, threshold)
    )

def _is_thumbs_up(landmarks: Sequence[Landmark]) -> bool:
    wrist = landmarks[WRIST]
    palm_h = _distance(wrist, landmarks[MIDDLE_PIP])
    threshold = palm_h * FINGER_EXTENSION_RATIO

    return (
        _thumb_extended(landmarks, wrist, threshold)
        and not _finger_extended(landmarks[INDEX_TIP], landmarks[INDEX_PIP], wrist, threshold)
        and not _finger_extended(landmarks[MIDDLE_TIP], landmarks[MIDDLE_PIP], wrist, threshold)
        and not _finger_extended(landmarks[RING_TIP], landmarks[RING_PIP], wrist, threshold)
        and not _finger_extended(landmarks[PINKY_TIP], landmarks[PINKY_PIP], wrist, threshold)
    )




GESTURES: list[tuple[str, Callable[[Sequence[Landmark]], bool]]] = [
    ("peace",       _is_peace),
    ("point",       _is_point),
    ("thumbs_up",   _is_thumbs_up),
    ("open_palm",   _is_open_palm),
    ("fist",        _is_fist),
]


def classify(landmarks: Sequence[Landmark]) -> str | None:
    """
    Return the gesture label for a single hand, or ``None`` if no
    registered gesture matches.
    """
    for name, check in GESTURES:
        if check(landmarks):
            return name
    return None
