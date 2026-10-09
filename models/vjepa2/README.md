# V-JEPA 2

[V-JEPA 2](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) predicts in
representation space, not pixels. So it is **scored** rather than shown in the video grid.
It's self-contained and doesn't use any other model folder.

**Surprise.**
1. The context half is encoded on its own, so the encoder never sees the future.
2. The predictor imagines the target half's features.
3. These are compared (L1) with the encoder's features of the real full clip, one value
   per 2-frame target step.

This is the V-JEPA intuitive-physics protocol. A copy-the-last-context-step baseline
gives the scale.

**Where V-JEPA puts the ball.** This uses the frozen evaluation decoder,
`eval_decoder.py`, which is the only readout allowed on V-JEPA features.
- It is a linear map from features to "ball in this cell", fitted only on *real* features
  of the context half (encoded alone).
- Its labels are the ball positions in those same frames.
- It never sees predictions, target-half labels or outcomes, so it **cannot learn the
  blockade**. That has to come from post-training the world model, and this decoder stays
  the same before and after.
- `tests/test_eval_isolation.py` enforces this (see `CLAUDE.md`).
- It still finds the ball beyond the plank in real frames: AUROC ~0.99.

To run it:
- `ball_probe.py`: videos for the 5-clip sample.
- `ball_probe_cv.py`: all clips, 5-fold, with run times.

Frames are resized to 256×256 rather than cropped, so the ball leaving through the
frame edge stays in view.

**Run it.** It runs locally: about 10 s per clip on an Apple M-series or A-series GPU,
and faster on CUDA.
```bash
pip install -e . -r models/vjepa2/requirements.txt
python models/vjepa2/run.py                              # surprise, all included clips (needs data/processed)
python models/vjepa2/run.py --sample data/eval/sample5   # the committed 5-clip sample
```

**Kinematic reference** (`kinematic.py`): fits a constant velocity to the tracked ball over the
last 1/3 s of context and extrapolates, hidden while its centre is under the plank or out of
frame. It is hard-coded (it learns nothing from outcomes), so it is a reference model, not an
evaluation tool. It is not wired into `ball_probe_cv.py` yet.

Encodings are cached in `outputs/vjepa2/cache/`, so reruns take seconds.

**Post-training** (`posttrain.py`, or `notebooks/colab_vjepa2_posttrain.ipynb` on Colab).
Only the predictor is trained; the encoder stays frozen, so the evaluation decoder is identical
before and after, and features are encoded once and cached.
```bash
python scripts/pack_clips.py                                 # on your computer, for Colab
python models/vjepa2/posttrain.py encode                     # cache encoder features
python models/vjepa2/posttrain.py train --run l1             # one predictor per CV fold
python models/vjepa2/posttrain.py train --run commit --loss commit   # "commit to a ball" (below)
python models/vjepa2/ball_probe_cv.py --predictor-run checkpoints/vjepa2/l1
```
`--loss commit` adds two label-free terms to the L1, so the predictor stops hedging towards "no
ball": tokens whose features change between steps, in the real future or the prediction, get more
weight; and an InfoNCE term makes each imagined future closer to its own real future than to other
training clips' futures. Each fold's predictor trains only on that fold's training clips (`cv_folds`), and
`ball_probe_cv.py` refuses a checkpoint that saw the clips it scores. Held-out prediction error
before and after, per outcome, is in `checkpoints/vjepa2/<run>/log.json`.

Targets are layer-normalised per token, the space V-JEPA 2's predictor was pretrained to output
(the Hugging Face encoder returns un-normalised features, about 5x larger). The evaluation decoder
reads every token after the same parameter-free layer norm, so real and imagined tokens are on one
scale. Predictions fed back in a rollout are rescaled to the last context step's per-token mean and spread.

Guards against overfitting and leakage: each fold's train and held-out clips may not share a clip
or a pass (`check_split`); 10% of the training clips (`--val-frac`) are held back and scored every
epoch, the best validation epoch's weights are kept (`--patience 3` stops early), and `log.json`
flags overfitting when validation L1 ends more than 2% above its best or is more than 1.25x the
L1 on training clips. The fold's held-out clips are never used to pick epochs. The experiment
summary lists flagged folds and kept epochs per run.

Two more options, each worth trying with and without:
- `--loss codes`: a spherical k-means codebook (`--codes 256`) is fitted on each fold's real
  training features (label-free; half the tokens are drawn from where features move). The
  predictor is trained with cross-entropy to pick each target token's code, and its output is
  snapped to the nearest code at inference, so it has to choose rather than average.
- `--rollout`: predict one step at a time, feeding the (detached, snapped if `codes`) prediction
  back in as context, both in training and at inference.

`experiments.py` runs the grid (plain, codes, rollout, codes_rollout), each before (`--epochs 0`,
the pretrained predictor in that variant's inference mode) and after, scores every run with
`ball_probe_cv.py`, and writes `outputs/vjepa2/experiments/summary.{md,json,png}`:
```bash
python models/vjepa2/experiments.py run --epochs 10
```
Headline metrics: P(correct outcome) and balanced accuracy over through/bounce/hidden, with each
outcome weighted equally (through passes outnumber the others), and the ball hit rate (imagined
ball within 48 px of the tracker's ball), median error and phantom-ball rate. The decoder on the
real future frames is the ceiling.

Threshold-free scores (in the summary first): they assume one ball and normalise each imagined
step's cell probabilities, so a blurred but well-placed prediction still counts: P(correct outcome)
and through-vs-blocked AUROC from the share of the ball beyond the plank, the most likely cell's
distance to the real ball, and ball cell AUROC. `enc_space_*` reads the same predictions mapped
into encoder space (gamma * prediction + beta, the encoder's own final layer norm) with the decoder
fitted on raw features, as a check on the layer-norm decoder. `scale_check.py` is a 16-clip
reproduction of the scale mismatch and its fix.

Unattended (overnight on Colab): `experiments.py run --resume --save-as overnight --push --branch <b>`
saves and pushes `results/<date>_<time>_overnight/` after every run, logs and skips a run that
fails, and with `--resume` skips runs already done (state in `outputs/vjepa2/experiments/grid.json`).

**Two-day plan.** `plan.py` runs the whole plan in `docs/experiment_plan.md` stage by stage (Colab:
`notebooks/colab_two_day_plan.ipynb`): baselines, the post-training grid, architecture options
(`posttrain.py --copy-gate`, `--hypotheses K`), longer and data-fraction runs, held-out generalisation
(`generalise.py`) and interpretation (`interpret.py`: change maps, layer patching). Progress is kept in
`outputs/vjepa2/plan/state.json`, so it resumes after a disconnect; after every step it pushes
`results/<date>_<time>_twoday/` and the status block in the plan file.

`python scripts/save_results.py --name <name> --push` saves a run's scores, figures and logs to
`results/<date>_<time>_<name>/` (never overwriting) and pushes them; the notebook does this at the
end. `results/README.md` indexes every saved run.

**Blocker figures.** `python models/vjepa2/blocker_figs.py` (after `ball_probe_cv.py`) asks whether
V-JEPA knows *where* the hidden blocker is. It fits nothing: it reads `per_clip.json` from
`ball_probe_cv.py` and places each pass by where its straight-line path from the context enters and
would leave the plank. P(blocked) is 1 − max P(ball beyond the plank) over the target.
- `blocker_map.png`: every pass's path, coloured by P(blocked), with the plank and the scene file's
  `blocker_polygon` outlined. Panels: true outcome, V-JEPA imagined, and a straight line plus the
  known blocker (a reference, not a model).
- `blocked_vs_crossing.png`: P(blocked) against the expected entry point and, separately, the
  expected exit point along the plank, with the blocker's span shaded.
- `crossing.json`: per-pass numbers and AUROCs.

For before/after, run it with `--label base`, then after post-training with
`--label fine-tuned --before outputs/vjepa2/blocker/crossing.json --out <new dir>`; the base curve
is drawn dashed.

**Predictive geometry.** `python models/vjepa2/geometry.py` renders, for each sample clip:
- the real and imagined token features, projected to colour with one shared PCA basis
- per-token change against V-JEPA's own output for an empty-scene clip
- `trajectories.png`: real versus imagined paths through feature space

**Encoder probing** (`probing.py`, CPU-friendly, resumable). Which readout of the frozen encoder
best locates the ball: encoder layer (4, 8, 12, 16, 20, output) x linear / small MLP x per-token /
3x3 neighbourhood / pooled over the frame x raw / layer-normed x per-cell sigmoid / one-ball softmax
(with a "no ball" option). Every probe is fitted like the evaluation decoder (real context half
encoded alone, same-frame labels, via `eval_decoder.examples_from`; `tests/test_eval_isolation.py`
covers it) and scored on held-out clips (`cv_folds`): context frames, real target frames (including
far-side cells, where no probe ever saw a ball) and, for the output layer, the pretrained predictor's
imagined target. It also runs the tracker + plank/blockade-rule reference (blockade fitted on outcomes)
and label-free blockade figures (P(blocked) against where the path crosses the plank, per-cell
distinctiveness of static context features, surprise by outcome).
```bash
python models/vjepa2/probing.py all          # encode, baseline, sweep, blockade, report, publish
```
Encodings take ~70 s a clip on a laptop CPU and ~80 MB a clip in `outputs/vjepa2/probe_cache/`;
results go to `outputs/vjepa2/probing/` and are saved to `results/<date>_<time>_vjepa-probing/`.

**Outputs** go to `outputs/vjepa2/`:
- `summary.json`: mean surprise curves per outcome (label-free)
- `per_clip.json`: per-clip surprise
- `features.npz`
