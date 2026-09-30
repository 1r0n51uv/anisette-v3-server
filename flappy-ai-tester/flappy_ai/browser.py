"""Controllo del gioco nel browser tramite Playwright.

Il player interagisce col gioco come un utente: legge i pixel del canvas e
invia input da tastiera/mouse. L'unica eccezione è la modalità lockstep del
clone locale, dove il tempo di gioco avanza su richiesta per addestrare
l'agente più veloce del tempo reale.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

GAME_DIR = Path(__file__).resolve().parent.parent / "game"


def clone_url(seed: int | None = None, lockstep: bool = False) -> str:
    q = []
    if seed is not None:
        q.append(f"seed={seed}")
    if lockstep:
        q.append("lockstep=1")
    return (GAME_DIR / "index.html").as_uri() + ("?" + "&".join(q) if q else "")


class GameBrowser:
    def __init__(self, url: str, headless: bool = True, canvas_selector: str = "canvas",
                 flap_key: str = "Space", lockstep: bool = False, record_video_dir: str | None = None):
        self.url = url
        self.lockstep = lockstep
        self.flap_key = flap_key
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        ctx_args = {"viewport": {"width": 480, "height": 640}}
        if record_video_dir:
            ctx_args["record_video_dir"] = record_video_dir
        self._context = self._browser.new_context(**ctx_args)
        self.page = self._context.new_page()
        self.page.goto(url, wait_until="load")
        self.canvas_selector = canvas_selector
        self.canvas = self.page.locator(canvas_selector).first
        self.canvas.wait_for(state="visible", timeout=30_000)
        self.page.mouse.click(1, 1)  # focus sulla pagina per ricevere i tasti
        self._fast_capture = True

    def reload(self) -> None:
        self.page.reload(wait_until="load")
        self.canvas.wait_for(state="visible", timeout=30_000)
        self.page.mouse.click(1, 1)

    # --- input reale ---
    def flap(self) -> None:
        self.page.keyboard.press(self.flap_key)

    def click_canvas(self) -> None:
        box = self.canvas.bounding_box()
        self.page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

    # --- tempo ---
    def advance(self, frames: int) -> None:
        """Lockstep: avanza di N frame. Tempo reale: attende N/60 s."""
        if self.lockstep:
            self.page.evaluate("n => window.__harness.advance(n)", frames)
        else:
            time.sleep(frames / 60)

    # --- percezione ---
    _READ_CANVAS = """([sel, frames]) => {
        if (frames > 0) window.__harness.advance(frames);
        try { return document.querySelector(sel).toDataURL('image/png').split(',')[1]; }
        catch (e) { return null; }
    }"""

    def screenshot(self, advance_frames: int = 0) -> bytes:
        """PNG di ciò che è disegnato sul canvas.

        ``toDataURL`` è ~15x più veloce di uno screenshot della pagina; se il canvas
        è "tainted" (immagini cross-origin) si ripiega sullo screenshot.
        In lockstep ``advance_frames`` fa avanzare il gioco nella stessa chiamata
        (un solo round-trip verso il browser); in tempo reale si attende.
        """
        if advance_frames and not self.lockstep:
            self.advance(advance_frames)
            advance_frames = 0
        if self._fast_capture:
            data = self.page.evaluate(self._READ_CANVAS, [self.canvas_selector, advance_frames])
            if data:
                return base64.b64decode(data)
            self._fast_capture = False  # i frame sono già stati avanzati dallo script
        elif advance_frames:
            self.advance(advance_frames)
        return self.canvas.screenshot(type="png", animations="allow", caret="initial")

    def ground_truth(self) -> dict | None:
        """Stato reale del clone, usato SOLO per misurare l'accuratezza della visione."""
        return self.page.evaluate("() => window.__harness ? window.__harness.state() : null")

    def close(self) -> None:
        try:
            self._context.close()
            self._browser.close()
        finally:
            self._pw.stop()
