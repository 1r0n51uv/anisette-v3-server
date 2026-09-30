"""Percezione black-box: estrae lo stato del gioco dai soli pixel del canvas.

Nessuna variabile interna del gioco viene letta. Uccello, tubi e terreno sono
riconosciuti con soglie di colore in HSV (OpenCV). Le soglie sono raccolte in un
``VisionProfile`` così che lo stesso codice funzioni sul clone locale e, dopo
calibrazione (vedi ``calibrate.py``), sul sito reale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class VisionProfile:
    # intervalli HSV in scala OpenCV (H 0-179, S 0-255, V 0-255)
    bird_lo: tuple[int, int, int] = (18, 150, 180)
    bird_hi: tuple[int, int, int] = (32, 255, 255)
    pipe_lo: tuple[int, int, int] = (35, 120, 100)
    pipe_hi: tuple[int, int, int] = (55, 255, 220)
    # quota del terreno come frazione dell'altezza del canvas
    ground_ratio: float = 400 / 512
    # l'uccello sta sempre nel terzo sinistro dello schermo
    bird_x_max_ratio: float = 0.40
    # pixel minimi per considerare valida una rilevazione
    min_bird_px: int = 60
    min_pipe_col_px: int = 40


PROFILES: dict[str, VisionProfile] = {
    "clone": VisionProfile(),
    # Valori di partenza per flappybird.io: vanno verificati con calibrate.py,
    # perché il sito non è raggiungibile dall'ambiente di sviluppo.
    "flappybird.io": VisionProfile(
        bird_lo=(15, 120, 170), bird_hi=(35, 255, 255),
        pipe_lo=(33, 100, 80), pipe_hi=(60, 255, 230),
        ground_ratio=0.79,
    ),
}


@dataclass
class Pipe:
    x0: int          # bordo sinistro (px)
    x1: int          # bordo destro (px)
    gap_top: int     # y del bordo inferiore del tubo alto
    gap_bottom: int  # y del bordo superiore del tubo basso


@dataclass
class Frame:
    width: int
    height: int
    ground_y: int
    bird: tuple[int, int, int, int] | None   # x0, y0, x1, y1
    pipes: list[Pipe] = field(default_factory=list)
    ground_strip: np.ndarray | None = None    # per rilevare se lo scenario scorre

    @property
    def bird_center(self) -> tuple[float, float] | None:
        if self.bird is None:
            return None
        x0, y0, x1, y1 = self.bird
        return (x0 + x1) / 2, (y0 + y1) / 2

    def next_pipe(self) -> Pipe | None:
        """Primo tubo non ancora superato dall'uccello."""
        if self.bird is None:
            return None
        bx0 = self.bird[0]
        ahead = [p for p in self.pipes if p.x1 >= bx0]
        return min(ahead, key=lambda p: p.x0) if ahead else None

    def pipe_after_next(self) -> Pipe | None:
        nxt = self.next_pipe()
        if nxt is None:
            return None
        later = [p for p in self.pipes if p.x0 > nxt.x1]
        return min(later, key=lambda p: p.x0) if later else None


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    """Intervalli [start, end) dove mask_1d è vero."""
    padded = np.concatenate([[0], mask_1d.astype(np.int8), [0]])
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def analyze(bgr: np.ndarray, profile: VisionProfile) -> Frame:
    h, w = bgr.shape[:2]
    ground_y = int(round(h * profile.ground_ratio))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    play = hsv[:ground_y]

    # --- uccello: blob giallo più grande nella parte sinistra ---
    bird = None
    xmax = int(w * profile.bird_x_max_ratio)
    bird_mask = cv2.inRange(play[:, :xmax], profile.bird_lo, profile.bird_hi)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bird_mask, connectivity=8)
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, bw, bh, area = stats[i]
        if area >= profile.min_bird_px:
            bird = (int(x), int(y), int(x + bw), int(y + bh))

    # --- tubi: colonne con molti pixel verdi sopra il terreno ---
    pipe_mask = cv2.inRange(play, profile.pipe_lo, profile.pipe_hi) > 0
    col_counts = pipe_mask.sum(axis=0)
    pipes: list[Pipe] = []
    for x0, x1 in _runs(col_counts >= profile.min_pipe_col_px):
        if x1 - x0 < 8:
            continue
        # il corpo del tubo (senza il bordo più largo) è la parte centrale
        core = pipe_mask[:, x0 + (x1 - x0) // 4: x1 - (x1 - x0) // 4]
        rows = core.mean(axis=1) > 0.5
        gaps = [(a, b) for a, b in _runs(~rows) if a > 0 and b < ground_y]
        if not gaps:
            continue
        gap_top, gap_bottom = max(gaps, key=lambda r: r[1] - r[0])
        pipes.append(Pipe(int(x0), int(x1), int(gap_top), int(gap_bottom)))

    strip = bgr[ground_y + 2: min(h, ground_y + 20)]
    return Frame(w, h, ground_y, bird, pipes, strip.copy() if strip.size else None)


def decode_png(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def debug_image(bgr: np.ndarray, frame: Frame) -> np.ndarray:
    """Sovrappone le rilevazioni al frame (utile per la tesi e la calibrazione)."""
    out = bgr.copy()
    cv2.line(out, (0, frame.ground_y), (frame.width, frame.ground_y), (0, 0, 255), 1)
    if frame.bird:
        x0, y0, x1, y1 = frame.bird
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 2)
    nxt = frame.next_pipe()
    for p in frame.pipes:
        color = (0, 0, 255) if p is nxt else (255, 0, 0)
        cv2.rectangle(out, (p.x0, p.gap_top), (p.x1, p.gap_bottom), color, 2)
    return out
