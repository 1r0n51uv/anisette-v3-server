# flappy-ai-tester

Player automatico per giochi in stile Flappy Bird, sviluppato per la tesi
**"L'utilizzo dell'AI per il testing online di applicativi videoludici"**.

Il player tratta il gioco come una **scatola nera**:

- **percepisce** solo i pixel del canvas (computer vision con OpenCV);
- **agisce** con input reali da tastiera o mouse (Playwright su Chromium);
- **decide** con due agenti a confronto: una baseline euristica e un agente di
  Reinforcement Learning (DQN);
- **registra** le metriche di partita (punteggio, causa della morte, latenza,
  frequenza di decisione) in CSV/JSON;
- **verifica** una soglia di punteggio: l'exit code è diverso da zero se la media
  non la raggiunge, quindi lo script si può usare in CI.

## Architettura

```
 ┌──────────── browser (Playwright/Chromium) ────────────┐
 │  gioco HTML5 (clone locale  oppure  flappybird.io)     │
 └───────┬───────────────────────────────▲───────────────┘
         │ pixel del canvas (PNG)        │ input: tasto SPAZIO
 ┌───────▼────────┐   osservazione  ┌────┴──────────────┐
 │ vision.py      │ ──────────────▶ │ agents.py         │
 │ OpenCV / HSV   │  6 feature      │ euristica | DQN   │
 └───────┬────────┘                 └───────────────────┘
         │ punteggio, game over, causa di morte
 ┌───────▼────────┐
 │ metrics.py     │ ──▶ results/*.csv, *_summary.json ──▶ plot_results.py
 └────────────────┘
```

| File | Ruolo |
|---|---|
| `game/index.html` | Clone di Flappy Bird usato come ambiente di test controllato |
| `flappy_ai/browser.py` | Apertura del gioco, input reali, cattura del canvas |
| `flappy_ai/vision.py` | Riconoscimento di uccello, tubi e terreno dai pixel; profili colore |
| `flappy_ai/env.py` | Ambiente Gymnasium: osservazione, ricompensa, game over, punteggio |
| `flappy_ai/agents.py` | Agente euristico, DQN e casuale |
| `flappy_ai/metrics.py` | Metriche per episodio e riepilogo |
| `play.py` | Fa giocare un agente e salva le metriche |
| `train.py` | Addestra il DQN (più browser in parallelo) |
| `calibrate.py` | Calibra le soglie di colore su un gioco reale |
| `plot_results.py` | Grafici per la tesi |
| `tests/` | Test pytest del gioco guidati dalla visione |

### Percezione (black-box)

Dal frame vengono estratti:

- l'**uccello**: il blob giallo più grande nel terzo sinistro dello schermo;
- i **tubi**: le colonne con molti pixel verdi; il varco è la sequenza più lunga di
  righe senza verde;
- il **terreno**: una striscia sotto la quota del suolo. Se tra due frame non cambia,
  lo scenario si è fermato, quindi è **game over**.

Anche il **punteggio** si ricava dalla visione: si conta il bordo destro di ogni tubo
che attraversa la linea dell'uccello. La **causa della morte** (terreno, soffitto,
tubo alto, tubo basso) si deduce dall'ultimo frame in cui l'uccello era vivo.

Sul clone, `window.__harness.state()` espone lo stato reale del gioco. Gli agenti
**non lo usano mai**: serve solo a misurare l'accuratezza della percezione
(`vision_score_accuracy`, `vision_death_cause_accuracy` nel riepilogo).

### Lockstep e tempo reale

Sul clone il gioco può girare in **lockstep** (`?lockstep=1`): il tempo avanza solo
quando l'agente ha deciso. In questo modo l'addestramento è più veloce del tempo
reale e le partite sono riproducibili (`?seed=N`). Sul sito reale si gioca per forza
in **tempo reale**. La differenza fra i due casi (latenza, frame persi) è un tema
interessante per la tesi: `play.py --realtime` la misura anche sul clone.

## Installazione

```bash
cd flappy-ai-tester
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Uso

```bash
# test automatici del gioco (clone, deterministici)
pytest -q

# baseline euristica: 20 partite, soglia sulla media di 50 tubi
python play.py --agent heuristic --episodes 20 --target-score 50

# addestramento DQN (~1 h su 4 core) e valutazione
python train.py --timesteps 300000 --envs 4
python play.py --agent rl --model models/dqn_flappy.zip --episodes 20 --target-score 50

# baseline casuale, per confronto
python play.py --agent random --episodes 20

# grafici
python plot_results.py --monitor runs/dqn --results results --out figures
```

Opzioni utili di `play.py`: `--headed` mostra il browser, `--video DIR` registra la
sessione, `--realtime` gioca il clone in tempo reale, `--max-score N` interrompe una
partita a N tubi (default 200, così un agente perfetto non gioca per sempre).

## Sul sito reale (flappybird.io)

L'ambiente di sviluppo non raggiunge flappybird.io, quindi il profilo
`flappybird.io` in `vision.py` contiene **valori di partenza da verificare**.
Procedura:

```bash
# 1. calibrazione: salva i frame con rilevazioni e maschere in calibration/
python calibrate.py --url https://flappybird.io --profile flappybird.io --headed

# 2. se uccello/tubi non sono riconosciuti, ritocca le soglie HSV del profilo
#    "flappybird.io" in flappy_ai/vision.py (e ground_ratio se la linea del terreno è sbagliata)

# 3. partite in tempo reale
python play.py --agent heuristic --url https://flappybird.io --profile flappybird.io --headed --episodes 5
python play.py --agent rl --model models/dqn_flappy.zip --url https://flappybird.io --profile flappybird.io --headed
```

Da tenere presente sul sito reale:

- il canvas può avere una risoluzione diversa dal clone. L'osservazione è
  normalizzata, ma l'agente euristico usa un margine in pixel riferito a 512 px di
  altezza e il DQN è addestrato sulla fisica del clone: aspettati un calo di
  prestazioni (in tesi è un risultato, non un difetto: *sim-to-real gap*);
- per riavviare dopo il game over, `FlappyEnv.reset` alterna barra spaziatrice e
  clic al centro del canvas. Se il pulsante "play" del sito è altrove, va adattato
  `click_canvas()` in `browser.py`;
- pubblicità o banner sopra il canvas possono disturbare la visione: usa
  `--canvas` in `calibrate.py` per puntare al canvas giusto.

## Metriche registrate

Per ogni partita (`results/<agente>_<data>.csv`):

| Campo | Significato |
|---|---|
| `score` / `true_score` | tubi superati secondo la visione / secondo il gioco (solo clone) |
| `death_cause` / `true_death_cause` | causa della morte stimata / reale |
| `steps`, `flaps` | decisioni prese, battiti d'ali |
| `duration_s`, `decisions_per_s` | durata della partita, frequenza di decisione |
| `latency_mean_ms`, `latency_p95_ms` | tempo input → cattura → analisi |
| `perception_misses` | frame in cui l'uccello non è stato trovato |
| `truncated` | partita interrotta al raggiungimento di `--max-score` |

Il riepilogo `*_summary.json` riporta media, mediana, min e max del punteggio,
distribuzione delle cause di morte, accuratezza della visione ed esito rispetto a
`--target-score`.
