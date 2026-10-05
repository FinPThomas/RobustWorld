# Results

One folder per saved run, named `<date>_<time>_<name>` (UTC); folders are never overwritten.
Each holds a README.md (when, which commit, config, headline table), the scores and
figures from `outputs/vjepa2/`, and training logs. Save a run with
`python scripts/save_results.py --name <name> [--note ...] [--push --branch <branch>]`;
the Colab notebook does this at the end. This index is rebuilt from the folders.

| run | headline |
|---|---|
| [2026-10-05_0425_vjepa-probing](2026-10-05_0425_vjepa-probing/README.md) | encoder probing, 104 probes: best far-side cell AUROC 0.9994 (L12-mlp-nbhd-ln-sigmoid), real-frame outcome AUROC 0.9996 |
