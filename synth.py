# synth type music note logic

from __future__ import annotations

import time
import numpy as np
import pygame.mixer

SAMPLE_RATE = 44100
VOLUME = 0.5
NOTE_DURATION_MS = 400
COOL_DOWN_MS = 300
FADE_OUT_MS = 50

_initalized = False

def _init() -> None:
    global _initalized
    if not _initalized:
        pygame.mixer.init(frequency=SAMPLE_RATE)
        _initalized = True

def _make_tone(frequency: float) -> np.ndarray:
    t = np.linspace(
        0,
        NOTE_DURATION_MS / 1000,
        int(SAMPLE_RATE * NOTE_DURATION_MS / 1000),
        endpoint=False,
        dtype=np.float32,
    )
    wave = np.sin(2 * np.pi * frequency * t)

    fade_samples = int(SAMPLE_RATE * FADE_OUT_MS / 1000)
    if fade_samples > 0 and len(wave) > fade_samples:
        envelope = np.ones_like(wave)
        envelope[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples)
        wave *= envelope

    wave *= VOLUME * 32767
    return wave.astype(np.int16)

class NoteSynth:
    def __init__(self) -> None:
        _init()
        self._last_gesture: str | None = None
        self._last_trigger_time: float = 0.0
        self._current_channel: pygame.mixer.Channel | None = None

    def play_for_gesture(self, gesture: str | None) -> None:
        GESTURE_NOTES = {
            "open_palm":  261.63,  # C4
            "fist":       329.63,  # E4
            "peace":      392.00,  # G4
            "point":      440.00,  # A4
            "thumbs_up":  523.25,  # C5
        }

        now = time.time()
        cooldown_sec = COOL_DOWN_MS / 1000

        #reset when no hand detected
        if gesture is None:
            self._last_gesture = None
            return

        #if same, don't play again
        if gesture == self._last_gesture:
            return

        #cool down check
        if now - self._last_trigger_time < cooldown_sec:
            return

        freq = GESTURE_NOTES.get(gesture)
        if freq is None:
            return

        wave = _make_tone(freq)
        sound = pygame.mixer.Sound(buffer=wave.tobytes())

        if self._current_channel is None:
            self._current_channel = pygame.mixer.Channel(0)
        self._current_channel.play(sound)

        self._last_gesture = gesture
        self._last_trigger_time = now


