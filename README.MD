# Turbofan Failure Detection — TCN vs. Autoencoder

## The Case

Can we detect that a jet engine is heading toward failure — before it actually happens?

Unplanned engine failure is expensive and, in the worst case, dangerous. If we can spot early
warning signs from the engine's own sensor data, maintenance can be scheduled proactively
instead of reactively.

## The Idea

We compare two different deep learning techniques, each asking a different question about the
same degradation process, rather than comparing two variants of the same technique:

- **TCN ("Was the model surprised?")** — predicts the next sensor reading from recent flight
  history. A large gap between prediction and reality signals a sudden, abrupt change in engine
  behaviour.
- **Autoencoder ("Does it recognize the whole picture?")** — tries to reconstruct an entire
  window of several flights from what it has memorized as "healthy". Failing to reconstruct it
  well signals that the overall pattern has gradually drifted away from normal.

We are **not** trying to find which technique is objectively best at predicting wear in general.
We're testing which of the two questions gives the clearest warning for *this specific kind* of
wear (gradual mechanical degradation, not sudden shocks).

Both models are trained **only** on the early, healthy cycles of each engine — neither model
ever sees a real failure during training. The alarm is simply the growing gap between what the
model expects/remembers as healthy and what it actually observes later in the engine's life.
This is the same core principle as the credit-card fraud example from the lecture (train only on
legitimate transactions).

## Dataset

**NASA Turbofan Jet Engine Degradation Simulation (C-MAPSS)** — simulated sensor data (not real
recorded flights), subset **FD001** (one operating condition, one fault mode).

Public mirror (no login required): https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
(also listed on Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

Each row is one operating **cycle** (roughly one flight). FD001 contains:
- **train_FD001.txt** — 100 engines, run from healthy to failure (variable length per engine,
  ranging from ~128 to ~360-plus cycles)
- **test_FD001.txt** — 100 engines, cut off before failure
- **RUL_FD001.txt** — the true "Remaining Useful Life" for each test engine at the point the
  recording was cut off — this is our ground-truth answer key

Each row has 26 columns: `unit_number`, `time_in_cycles`, 3 operational settings, and 21 sensor
measurements.

## Fair Comparison

TCN's prediction error and the Autoencoder's reconstruction error are **not** on the same scale
and can't be compared directly. Instead, both models are evaluated at the same cycles, on the
same test engines, against the same ground truth (true RUL), using two shared metrics:

- **Correlation** — how strongly each alarm signal correlates with falling RUL
- **Lead time** — how many cycles before failure each alarm signal crossed a warning threshold

## Common Settings

So that everyone's models stay comparable:

- Subset **FD001** only (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`)
- Same sensor selection (drop the ~7 sensors that are effectively constant)
- Same window size (e.g. 30 cycles) for both TCN's history window and the Autoencoder's window
- Same definition of "healthy period" (e.g. the first ~30% of each training engine's life)

## Dataset Setup

Data is not committed to this repository (`data/raw/` is gitignored). Each team member should:

1. Download the dataset from the link above (or from Kaggle)
2. Extract it so the three FD001 files end up at:

       data/raw/CMAPSSData/train_FD001.txt
       data/raw/CMAPSSData/test_FD001.txt
       data/raw/CMAPSSData/RUL_FD001.txt

## Team & Presentation Split

- **Emmy** — slides 1–5: introduction + TCN model
- **Mahtab** — slides 6–10: Autoencoder model
- **Spirit** — slides 11–15: connects both models back to the lecture's architecture diagrams,
  explains how they work at a higher level, and closes with business value / conclusion

## Git Workflow

`dev` is the shared branch. Each person branches off `dev` into their own `feat/<name>` branch
and owns one notebook in `notebooks/`, to avoid merge conflicts.
