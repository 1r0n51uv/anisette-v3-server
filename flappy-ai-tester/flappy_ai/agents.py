"""Agenti: baseline euristica e agente RL (DQN, stable-baselines3)."""
from __future__ import annotations

import numpy as np

H = 512  # altezza del canvas, per riportare in pixel l'osservazione normalizzata


class HeuristicAgent:
    """Controllore a regole: sbatte le ali quando l'uccello scende troppo vicino
    al bordo inferiore del varco del prossimo tubo."""

    name = "heuristic"

    def __init__(self, margin_px: float = 24.0):
        self.margin = margin_px / H

    def act(self, obs: np.ndarray) -> int:
        vy, below_bottom = obs[1] * 10, obs[4]
        # distanza tra il centro dell'uccello e il bordo basso del varco
        if below_bottom < self.margin and vy > -2:
            return 1
        return 0


class RLAgent:
    name = "rl"

    def __init__(self, model_path: str):
        from stable_baselines3 import DQN
        self.model = DQN.load(model_path, device="cpu")

    def act(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


class RandomAgent:
    name = "random"

    def __init__(self, flap_prob: float = 0.08, seed: int = 0):
        self.p = flap_prob
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.random() < self.p)
