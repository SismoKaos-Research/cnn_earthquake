# Ground motion

Peak ground velocity/acceleration from a short window, after the Nurtas et
al. replication. Baselines are run before any network, in
`groundmotion_baselines.py`.

## Scripts

- **`cnn_groundmotion.py`** — Peak ground motion from a 3-second window: the network, against the floor.
- **`groundmotion_baselines.py`** — Non-neural floors for the peak-ground-motion task, run BEFORE any network.
- **`groundmotion_summary.py`** — Collate the ground-motion experiment grid into one table.

## Running

Shared code lives in `seismolib`, installed editable (`uv pip install -e .`),
so scripts run from anywhere:

```bash
python3 src/groundmotion/<script>.py --help
```

Project-wide results spanning several families are in [`docs/`](../../docs/):
`report.md` (the full write-up) and `accuracy_summary.md` (headline numbers per task).
