"""Grafici per la tesi: curva di apprendimento e confronto fra agenti.

    python plot_results.py --monitor runs/dqn --results results --out figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def learning_curve(monitor_dir: Path, out: Path) -> None:
    frames = []
    for f in sorted(monitor_dir.glob("*.monitor.csv")):
        df = pd.read_csv(f, skiprows=1)  # la prima riga è un header JSON di SB3
        frames.append(df)
    if not frames:
        print(f"nessun monitor.csv in {monitor_dir}")
        return
    df = pd.concat(frames).sort_values("t").reset_index(drop=True)
    df["steps"] = df["l"].cumsum()
    df["score_ma"] = df["score"].rolling(100, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["steps"], df["score"], alpha=0.2, lw=0.8, label="punteggio per episodio")
    ax.plot(df["steps"], df["score_ma"], lw=2, label="media mobile (100 episodi)")
    ax.set_xlabel("passi di addestramento")
    ax.set_ylabel("tubi superati")
    ax.set_title("DQN: curva di apprendimento")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=150)
    print(f"salvato {out / 'learning_curve.png'}")


def agent_comparison(results_dir: Path, out: Path) -> None:
    files = sorted(results_dir.glob("*.csv"))
    if not files:
        print(f"nessun CSV in {results_dir}")
        return
    df = pd.concat(pd.read_csv(f) for f in files)
    agents = sorted(df["agent"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot([df[df.agent == a]["score"] for a in agents], tick_labels=agents)
    axes[0].set_ylabel("tubi superati")
    axes[0].set_title("Distribuzione dei punteggi")
    causes = (df[~df.truncated].groupby(["agent", "death_cause"]).size().unstack(fill_value=0))
    if not causes.empty:
        causes.plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_title("Cause di morte (stimate dalla visione)")
    axes[1].set_xlabel("")
    fig.tight_layout()
    fig.savefig(out / "agent_comparison.png", dpi=150)
    print(f"salvato {out / 'agent_comparison.png'}")
    print(df.groupby("agent")[["score", "latency_mean_ms", "decisions_per_s"]].describe().T.round(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--monitor", default="runs/dqn")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    learning_curve(Path(args.monitor), out)
    agent_comparison(Path(args.results), out)


if __name__ == "__main__":
    main()
