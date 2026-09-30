"""Calibrazione del profilo di visione su un gioco reale (es. flappybird.io).

Apre il gioco, avvia una partita, salva alcuni frame con le rilevazioni
sovrapposte e le maschere di colore. Guardando le immagini si capisce se le
soglie HSV del profilo vanno ritoccate in flappy_ai/vision.py.

    python calibrate.py --url https://flappybird.io --profile flappybird.io --headed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from flappy_ai.browser import GameBrowser, clone_url
from flappy_ai.vision import PROFILES, analyze, debug_image, decode_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--profile", choices=list(PROFILES), default="clone")
    ap.add_argument("--canvas", default="canvas", help="selettore CSS del canvas di gioco")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--out", default="calibration")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = PROFILES[args.profile]
    b = GameBrowser(args.url or clone_url(seed=1), headless=not args.headed, canvas_selector=args.canvas)
    try:
        b.click_canvas()
        for i in range(12):
            if i % 2 == 0:
                b.flap()
            b.advance(12)
            img = decode_png(b.screenshot())
            f = analyze(img, prof)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            bird = cv2.inRange(hsv, prof.bird_lo, prof.bird_hi)
            pipe = cv2.inRange(hsv, prof.pipe_lo, prof.pipe_hi)
            masks = np.hstack([cv2.cvtColor(bird, cv2.COLOR_GRAY2BGR), cv2.cvtColor(pipe, cv2.COLOR_GRAY2BGR)])
            cv2.imwrite(str(out / f"frame{i:02d}.png"), np.hstack([img, debug_image(img, f), masks]))
            print(f"frame {i:02d}: uccello={f.bird} tubi={[(p.x0, p.gap_top, p.gap_bottom) for p in f.pipes]}")
    finally:
        b.close()
    print(f"Immagini in {out}/: originale | rilevazioni | maschera uccello | maschera tubi")


if __name__ == "__main__":
    main()
