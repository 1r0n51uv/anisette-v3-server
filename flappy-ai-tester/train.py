"""Addestra l'agente DQN sul clone locale (lockstep, più browser in parallelo).

La curva di apprendimento (punteggio per episodio) viene salvata nei file
``monitor.csv`` sotto --logdir, pronta per i grafici della tesi.

    python train.py --timesteps 300000 --envs 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


def make_env(rank: int, logdir: str, max_score: int, shaping: float):
    def _init():
        from flappy_ai.browser import clone_url
        from flappy_ai.env import FlappyEnv
        from flappy_ai.vision import PROFILES
        env = FlappyEnv(clone_url(seed=1000 + rank, lockstep=True), PROFILES["clone"],
                        lockstep=True, max_score=max_score, shaping=shaping)
        return Monitor(env, str(Path(logdir) / f"env{rank}"), info_keywords=("score",))
    return _init


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--max-score", type=int, default=100, help="tetto per episodio durante il training")
    ap.add_argument("--shaping", type=float, default=0.1,
                    help="bonus per l'allineamento al varco (0 = solo ricompensa del gioco)")
    ap.add_argument("--logdir", default="runs/dqn")
    ap.add_argument("--out", default="models/dqn_flappy")
    ap.add_argument("--resume", help="modello .zip da cui riprendere")
    args = ap.parse_args()

    torch.set_num_threads(1)  # la CPU serve ai browser; la rete è piccola
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    venv = SubprocVecEnv([make_env(i, args.logdir, args.max_score, args.shaping) for i in range(args.envs)])
    if args.resume:
        model = DQN.load(args.resume, env=venv, device="cpu")
    else:
        model = DQN(
            "MlpPolicy", venv, device="cpu", seed=0, verbose=1,
            policy_kwargs={"net_arch": [64, 64]},
            learning_rate=5e-4, buffer_size=100_000, learning_starts=5_000,
            batch_size=64, gamma=0.99, train_freq=1, gradient_steps=2,
            target_update_interval=2_000,
            exploration_fraction=0.2, exploration_initial_eps=0.2, exploration_final_eps=0.0,
            tensorboard_log=None,
        )
    ckpt = CheckpointCallback(save_freq=max(1, 50_000 // args.envs), save_path=args.logdir, name_prefix="dqn")
    try:
        model.learn(total_timesteps=args.timesteps, callback=ckpt, log_interval=20,
                    reset_num_timesteps=not args.resume)
    finally:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out)
        venv.close()
    print(f"Modello salvato in {args.out}.zip")


if __name__ == "__main__":
    main()
