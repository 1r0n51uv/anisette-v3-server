"""Ambiente Gymnasium che avvolge il gioco nel browser.

Osservazione (6 valori normalizzati, tutti ricavati dai pixel):
    0  quota dell'uccello                         y / H
    1  velocità verticale (px/frame)              vy / 10
    2  distanza orizzontale dal prossimo tubo     dx / W
    3  bordo alto del varco, relativo all'uccello  (gap_top - y) / H
    4  bordo basso del varco, relativo             (gap_bottom - y) / H
    5  centro del varco successivo, relativo       (gap2_center - y) / H
Azioni: 0 = non fare nulla, 1 = sbattere le ali.
Ricompensa: +0.1 per passo in vita, +1 per tubo superato, -1 alla morte.
Con ``shaping > 0`` (solo in addestramento) si aggiunge un bonus denso, fino a
``shaping`` per passo, quando l'uccello è allineato al centro del varco: senza,
la ricompensa per tubo è troppo rara e il DQN impara solo a sopravvivere fino
al primo tubo.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from .browser import GameBrowser
from .vision import Frame, Pipe, VisionProfile, analyze, decode_png


@dataclass
class EpisodeStats:
    score: int = 0
    steps: int = 0
    flaps: int = 0
    started: float = field(default_factory=time.perf_counter)
    latencies_ms: list[float] = field(default_factory=list)
    perception_misses: int = 0
    death_cause: str | None = None


class FlappyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, url: str, profile: VisionProfile, lockstep: bool = False,
                 frame_skip: int = 2, headless: bool = True, max_score: int | None = None,
                 static_frames_for_death: int = 2, record_video_dir: str | None = None,
                 shaping: float = 0.0):
        super().__init__()
        self.profile = profile
        self.frame_skip = frame_skip
        self.max_score = max_score
        self.static_needed = static_frames_for_death
        self.shaping = shaping
        self.browser = GameBrowser(url, headless=headless, lockstep=lockstep,
                                   record_video_dir=record_video_dir)
        self.observation_space = gym.spaces.Box(-2.0, 2.0, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

        self.ref_x: float | None = None   # bordo sinistro dell'uccello (fisso in Flappy Bird)
        self.frame: Frame | None = None
        self.prev_frame: Frame | None = None
        self.last_alive_frame: Frame | None = None
        self.bird_y: float = 0.0
        self.bird_vy: float = 0.0
        self.static_count = 0
        self.stats = EpisodeStats()
        self._last_obs = np.zeros(6, np.float32)
        self._in_progress = False
        self._t_last_capture = time.perf_counter()

    # ------------------------------------------------------------------ percezione
    def _capture(self, advance_frames: int = 0) -> Frame:
        frame = analyze(decode_png(self.browser.screenshot(advance_frames)), self.profile)
        self.prev_frame, self.frame = self.frame, frame
        return frame

    def _ground_moving(self) -> bool:
        a, b = self.prev_frame, self.frame
        if a is None or b is None or a.ground_strip is None or b.ground_strip is None:
            return True
        return bool(np.any(a.ground_strip != b.ground_strip))

    def _next_pipes(self, frame: Frame) -> tuple[Pipe | None, Pipe | None]:
        ahead = sorted((p for p in frame.pipes if p.x1 >= self.ref_x), key=lambda p: p.x0)
        return (ahead[0] if ahead else None), (ahead[1] if len(ahead) > 1 else None)

    def _count_passed(self) -> int:
        """Tubi che hanno attraversato la linea dell'uccello tra i due ultimi frame."""
        if self.prev_frame is None or self.frame is None:
            return 0
        passed = 0
        for old in self.prev_frame.pipes:
            if old.x1 < self.ref_x:
                continue
            for new in self.frame.pipes:
                if 0 <= old.x1 - new.x1 <= 60 and new.x1 < self.ref_x:
                    passed += 1
                    break
        return passed

    def _observe(self) -> np.ndarray:
        f = self.frame
        H, W = f.height, f.width
        if f.bird_center is None:
            self.stats.perception_misses += 1
            return self._last_obs
        y = f.bird_center[1]
        now = time.perf_counter()
        # velocità in px per frame di gioco (in tempo reale: frame stimati dal tempo trascorso)
        frames_elapsed = self.frame_skip if self.browser.lockstep else max(1.0, (now - self._t_last_capture) * 60)
        self.bird_vy = (y - self.bird_y) / frames_elapsed
        self.bird_y = y
        self._t_last_capture = now

        nxt, after = self._next_pipes(f)
        if nxt is None:
            dx, gt, gb = W - self.ref_x, f.ground_y * 0.35, f.ground_y * 0.65
        else:
            dx, gt, gb = nxt.x0 - self.ref_x, nxt.gap_top, nxt.gap_bottom
        g2 = (after.gap_top + after.gap_bottom) / 2 if after else (gt + gb) / 2
        obs = np.array([y / H, self.bird_vy / 10, dx / W, (gt - y) / H, (gb - y) / H, (g2 - y) / H],
                       dtype=np.float32)
        self._last_obs = np.clip(obs, -2, 2)
        return self._last_obs

    def _infer_death_cause(self) -> str:
        f = self.last_alive_frame
        if f is None or f.bird is None:
            return "unknown"
        x0, y0, x1, y1 = f.bird
        if y1 >= f.ground_y - 6:
            return "ground"
        if y0 <= 2:
            return "ceiling"
        for p in f.pipes:
            if x1 >= p.x0 - 6 and x0 <= p.x1:
                return "pipe_top" if (y0 + y1) / 2 < (p.gap_top + p.gap_bottom) / 2 else "pipe_bottom"
        return "unknown"

    # ------------------------------------------------------------------ API Gymnasium
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        b = self.browser
        if self._in_progress:
            # episodio precedente troncato: la partita è ancora viva, si ricomincia da zero
            b.reload()
            self.frame = None
        # 1) se siamo al game over (scenario fermo), premi finché il gioco riparte
        self._capture()
        for attempt in range(100):
            self._capture(2)
            if self._ground_moving() and self.frame.bird is not None:
                break
            # alcuni giochi ripartono solo con un clic (pulsante "play"), non col tasto
            if attempt % 4 == 3:
                b.click_canvas()
            else:
                b.flap()
            b.advance(10)
        else:
            raise RuntimeError("Impossibile (ri)avviare il gioco: controlla il profilo di visione")
        if self.ref_x is None:
            x0, _, x1, _ = self.frame.bird
            self.ref_x = float(x0)
        # 2) schermata "get ready": il primo battito d'ali avvia la partita
        self.bird_y = self.frame.bird_center[1]
        b.flap()
        self._capture(self.frame_skip)
        self.static_count = 0
        self.stats = EpisodeStats()
        self.last_alive_frame = self.frame
        self._t_last_capture = time.perf_counter()
        self._in_progress = True
        return self._observe(), {}

    def step(self, action: int):
        t0 = time.perf_counter()
        if action == 1:
            self.browser.flap()
            self.stats.flaps += 1
        self._capture(self.frame_skip)
        obs = self._observe()
        self.stats.steps += 1

        if self._ground_moving():
            self.static_count = 0
            self.last_alive_frame = self.frame
        else:
            self.static_count += 1
        dead = self.static_count >= self.static_needed

        passed = 0 if dead else self._count_passed()
        self.stats.score += passed
        reward = -1.0 if dead else 0.1 + passed
        if self.shaping and not dead:
            gap_center_rel = (obs[3] + obs[4]) / 2
            reward += self.shaping * max(0.0, 1.0 - abs(float(gap_center_rel)) / 0.1)
        truncated = self.max_score is not None and self.stats.score >= self.max_score
        if dead:
            self.stats.death_cause = self._infer_death_cause()
            self._in_progress = False
        self.stats.latencies_ms.append((time.perf_counter() - t0) * 1000)
        return obs, reward, dead, truncated, {"score": self.stats.score}

    def close(self):
        self.browser.close()
