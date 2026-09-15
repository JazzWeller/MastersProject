#!/usr/bin/env python3
"""Synthesizes the handful of short sound effects the GUI uses, straight
from Python's standard library (`wave` + `math`), so there's nothing to
download and no licensing question -- these are original, generated here.

Run once: `python tools/generate_sounds.py`. Writes 16-bit mono WAV files
into `assets/sounds/`; `gui/assets.py` picks up anything it finds there by
filename (see AssetCache._load_sounds), so dropping in nicer replacements
later needs no code change.
"""

from __future__ import annotations

import math
import os
import random
import struct
import wave

SR = 22050
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "sounds")


def _write(name: str, samples) -> None:
    path = os.path.join(OUT_DIR, f"{name}.wav")
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        frames = b"".join(struct.pack("<h", max(-32767, min(32767, int(s)))) for s in samples)
        f.writeframes(frames)
    print("wrote", path, f"({len(samples) / SR * 1000:.0f}ms)")


def _envelope(i: int, n: int, attack: float, release: float) -> float:
    a = int(n * attack)
    r = int(n * release)
    if i < a and a > 0:
        return i / a
    if i > n - r and r > 0:
        return max(0.0, (n - i) / r)
    return 1.0


def tone(freq: float, dur_ms: float, attack=0.05, release=0.6, amp=0.5, wave_fn=math.sin) -> list:
    n = int(SR * dur_ms / 1000)
    out = []
    for i in range(n):
        t = i / SR
        env = _envelope(i, n, attack, release)
        out.append(wave_fn(2 * math.pi * freq * t) * amp * env * 32767)
    return out


def chord(freqs, dur_ms, attack=0.05, release=0.6, amp=0.35) -> list:
    parts = [tone(f, dur_ms, attack, release, amp) for f in freqs]
    return [sum(vals) / len(vals) for vals in zip(*parts)]


def noise(dur_ms: float, attack=0.02, release=0.7, amp=0.4, seed=0) -> list:
    rng = random.Random(seed)
    n = int(SR * dur_ms / 1000)
    return [rng.uniform(-1, 1) * amp * _envelope(i, n, attack, release) * 32767 for i in range(n)]


def mix(*tracks) -> list:
    n = max(len(t) for t in tracks)
    out = [0.0] * n
    for t in tracks:
        for i, v in enumerate(t):
            out[i] += v
    peak = max(1.0, max(abs(v) for v in out) / 32767)
    return [v / peak for v in out]


def sequence(*tracks) -> list:
    out = []
    for t in tracks:
        out.extend(t)
    return out


def silence(dur_ms: float) -> list:
    return [0.0] * int(SR * dur_ms / 1000)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    _write("click", tone(880, 45, attack=0.02, release=0.7, amp=0.35))

    _write(
        "card_move",
        mix(tone(520, 70, attack=0.1, release=0.7, amp=0.22), noise(60, attack=0.1, release=0.8, amp=0.10)),
    )

    _write(
        "damage",
        mix(tone(110, 160, attack=0.02, release=0.85, amp=0.55, wave_fn=math.sin), noise(90, attack=0.0, release=0.8, amp=0.18)),
    )

    _write(
        "destroy",
        sequence(
            mix(tone(180, 130, attack=0.0, release=0.6, amp=0.4), noise(220, attack=0.0, release=0.85, amp=0.32)),
        ),
    )

    _write(
        "key_forge",
        sequence(
            tone(523.25, 110, attack=0.02, release=0.5, amp=0.3),  # C5
            tone(659.25, 110, attack=0.02, release=0.5, amp=0.3),  # E5
            tone(783.99, 110, attack=0.02, release=0.5, amp=0.3),  # G5
            chord([1046.50, 1318.51], 260, attack=0.02, release=0.75, amp=0.3),  # C6+E6
        ),
    )

    _write("gain", tone(1568.0, 90, attack=0.01, release=0.75, amp=0.28, wave_fn=math.sin))

    print("done.")


if __name__ == "__main__":
    main()
