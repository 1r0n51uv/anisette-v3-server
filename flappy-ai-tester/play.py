"""Fa giocare un agente e registra le metriche di partita.

Esempi:
    python play.py --agent heuristic --episodes 20
    python play.py --agent rl --model models/dqn_flappy.zip --episodes 20 --target-score 50
    python play.py --agent heuristic --url https://flappybird.io --profile flappybird.io --headed
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from flappy_ai.agents import HeuristicAgent, RandomAgent, RLAgent
from flappy_ai.browser import clone_url
from flappy_ai.env import FlappyEnv
from flappy_ai.metrics import EpisodeRecord, MetricsWriter, percentile
from flappy_ai.vision import PROFILES


def build_agent(args):
    if args.agent == "heuristic":
        return HeuristicAgent()
    if args.agent == "random":
        return RandomAgent(seed=args.seed)
    if not args.model:
        sys.exit("--model è obbligatorio con --agent rl")
    return RLAgent(args.model)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", choices=["heuristic", "rl", "random"], default="heuristic")
    ap.add_argument("--model", help="modello DQN (.zip) per --agent rl")
    ap.add_argument("--url", help="URL del gioco (default: clone locale)")
    ap.add_argument("--profile", choices=list(PROFILES), default="clone")
    ap.add_argument("--realtime", action="store_true",
                    help="sul clone: gioca in tempo reale invece che in lockstep")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-score", type=int, default=200, help="interrompe la partita a questo punteggio")
    ap.add_argument("--target-score", type=float, default=None,
                    help="soglia sulla media: exit code 1 se non raggiunta (utile in CI)")
    ap.add_argument("--headed", action="store_true", help="mostra il browser")
    ap.add_argument("--video", help="cartella dove salvare il video della sessione")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    on_clone = args.url is None
    lockstep = on_clone and not args.realtime
    url = clone_url(seed=args.seed, lockstep=lockstep) if on_clone else args.url

    agent = build_agent(args)
    env = FlappyEnv(url, PROFILES[args.profile], lockstep=lockstep, headless=not args.headed,
                    max_score=args.max_score, record_video_dir=args.video)
    target = "clone" if on_clone else url
    writer = MetricsWriter(args.out, f"{agent.name}_{time.strftime('%Y%m%d_%H%M%S')}")
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = truncated = False
            while not (done or truncated):
                obs, _, done, truncated, _ = env.step(agent.act(obs))
            s = env.stats
            truth = env.browser.ground_truth() if on_clone else None
            duration = time.perf_counter() - s.started
            rec = EpisodeRecord(
                episode=ep, agent=agent.name, target=target, score=s.score,
                true_score=truth["score"] if truth else None,
                death_cause=None if truncated else s.death_cause,
                true_death_cause=truth["deathCause"] if truth else None,
                steps=s.steps, flaps=s.flaps, duration_s=round(duration, 3),
                decisions_per_s=round(s.steps / duration, 1) if duration else 0.0,
                latency_mean_ms=round(sum(s.latencies_ms) / max(1, len(s.latencies_ms)), 2),
                latency_p95_ms=round(percentile(s.latencies_ms, 0.95), 2),
                perception_misses=s.perception_misses, truncated=truncated,
            )
            writer.add(rec)
            print(f"[{agent.name}] partita {ep + 1}/{args.episodes}: punteggio {s.score}"
                  f" (reale {rec.true_score}), causa {rec.death_cause} (reale {rec.true_death_cause})")
    finally:
        env.close()

    summary = writer.summary(args.target_score)
    print(json.dumps(summary, indent=2))
    print(f"CSV: {writer.csv_path}")
    return 0 if summary.get("passed", True) else 1


if __name__ == "__main__":
    sys.exit(main())
