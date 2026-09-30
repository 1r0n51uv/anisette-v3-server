"""Raccolta delle metriche di partita in CSV/JSON."""
from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EpisodeRecord:
    episode: int
    agent: str
    target: str
    score: int                 # punteggio stimato dalla visione
    true_score: int | None     # solo sul clone: verità di riferimento
    death_cause: str | None    # causa stimata dalla visione
    true_death_cause: str | None
    steps: int
    flaps: int
    duration_s: float
    decisions_per_s: float
    latency_mean_ms: float
    latency_p95_ms: float
    perception_misses: int
    truncated: bool


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


class MetricsWriter:
    FIELDS = list(EpisodeRecord.__dataclass_fields__)

    def __init__(self, out_dir: str | Path, run_name: str | None = None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run = run_name or time.strftime("run_%Y%m%d_%H%M%S")
        self.csv_path = self.dir / f"{self.run}.csv"
        self.records: list[EpisodeRecord] = []
        with self.csv_path.open("w", newline="") as f:
            csv.DictWriter(f, self.FIELDS).writeheader()

    def add(self, rec: EpisodeRecord) -> None:
        self.records.append(rec)
        with self.csv_path.open("a", newline="") as f:
            csv.DictWriter(f, self.FIELDS).writerow(asdict(rec))

    def summary(self, target_score: float | None = None) -> dict:
        r = self.records
        scores = [x.score for x in r]
        causes: dict[str, int] = {}
        for x in r:
            if not x.truncated:
                causes[x.death_cause or "unknown"] = causes.get(x.death_cause or "unknown", 0) + 1
        with_truth = [x for x in r if x.true_score is not None]
        summary = {
            "run": self.run,
            "episodes": len(r),
            "score_mean": statistics.fmean(scores) if scores else 0,
            "score_median": statistics.median(scores) if scores else 0,
            "score_max": max(scores, default=0),
            "score_min": min(scores, default=0),
            "death_causes": causes,
            "latency_mean_ms": statistics.fmean([x.latency_mean_ms for x in r]) if r else 0,
            "latency_p95_ms": max((x.latency_p95_ms for x in r), default=0),
            "decisions_per_s": statistics.fmean([x.decisions_per_s for x in r]) if r else 0,
        }
        if with_truth:
            summary["vision_score_accuracy"] = sum(x.score == x.true_score for x in with_truth) / len(with_truth)
            died = [x for x in with_truth if not x.truncated and x.true_death_cause]
            if died:
                summary["vision_death_cause_accuracy"] = (
                    sum(x.death_cause == x.true_death_cause for x in died) / len(died))
        if target_score is not None:
            summary["target_score"] = target_score
            summary["passed"] = summary["score_mean"] >= target_score
        (self.dir / f"{self.run}_summary.json").write_text(json.dumps(summary, indent=2))
        return summary
