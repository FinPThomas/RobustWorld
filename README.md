# RobustWorld

Turns overhead videos of a ball rolling under an occluder into short clips for
world-model training and evaluation (Wan 2.2, V-JEPA 2). Each clip is split in
half: the model sees the **context** (ball rolling in) and must predict the
**target** (ball comes through, bounces back, or stays hidden behind a blockade).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

ffmpeg comes bundled with `imageio-ffmpeg`, so no system install is needed.

## Processing videos

```bash
.venv/bin/robustworld data/raw/videos/              # every .mp4 in the folder
.venv/bin/robustworld data/raw/videos/start.mp4     # one video
.venv/bin/robustworld start.mp4 --from track        # force a rerun from a stage
```

Stages are cached: each one reruns only if its inputs changed (new video, edited
scene file) or `--from` forces it.

| Stage      | Output                                                        |
|------------|---------------------------------------------------------------|
| `downsize` | `data/interim/<name>/<name>_512.mp4`: 512x512, source fps, no audio |
| `track`    | `data/interim/<name>/track.csv`: ball position and hand signal for every frame; `backgrounds.npz` |
| `passes`   | `data/interim/<name>/passes.json`: one record per approach to the occluder |
| `clips`    | `data/processed/clips/<name>/`: clips, hand masks, `manifest.jsonl`, `sheets/`, `review.jpg` |

All videos' manifests are merged into `data/processed/clips/manifest.jsonl`.

### Adding a new video

1. Put the `.mp4` in `data/raw/videos/` and run `robustworld` on it.
2. The first run downsizes the video, then stops and writes:
   - `data/interim/<name>/calibration.png`: the first frame with a pixel grid
   - `configs/scenes/<name>.json`: a template scene file
3. In the scene file, set `occluder_polygon` to the occluder's outline in (x, y)
   pixels, then delete the `"TODO"` line. Optionally add `ignore_regions` for
   clutter such as furniture that shifts during the recording. See
   `configs/scenes/start.json`.
4. Run it again.

The defaults assume a fixed camera and a **blue** ball. A different ball colour
needs `ball_hsv_lo`/`ball_hsv_hi` set in the scene file (OpenCV HSV, H 0–179).
Moments where the scene changes for good (a prop nudged, the chair moved) are
detected automatically. To override them, set `background_breaks_s`.

## Clip format

- 48 frames, 16 fps (3 s), 512x512, H.264.
- **Frames 0–23: context.** Frame 23 is the last frame before the ball's
  outline touches the occluder.
- **Frames 24–47: target.**
- For V-JEPA 2, context and target are 24 frames each. For Wan (frame
  counts of 4n+1), condition on the context and generate 25 frames starting
  from frame 23.

Every outcome is kept, including the ball staying hidden: the blockade under the
occluder is part of what the model has to learn. A clip is marked
`include: false` only if:

- the context half is unusable: another ball is in it, or the ball didn't roll
  in from the frame edge, or
- a hand touches the ball after the throw. The ball must roll untouched for the
  last 0.5 s of context and the whole target.

A hand elsewhere in the frame doesn't exclude the clip. It is saved instead as
`<clip>_hand.npz` (`masks`: bool `[48, 512, 512]`), so it can be masked out of
the loss or filled in from the background later.

Main `manifest.jsonl` fields: `clip_id`, `path`, `include`, `reasons`, `outcome`
(`through` / `hidden` / `bounce` within the clip), `ball_at_start`,
`entry_frame`, `occlusion_start_frame`, `hidden_frame`, `reappear_frame`,
`exit_frame` (clip frame indices, or null when outside the clip),
`hand_mask_path`, `hand_in_context`, `hand_in_target`, `source_video`,
`source_frames`.

To review the clips, open `review.jpg`, which has one row per clip. Hand masks
are shown in magenta, and a red bar marks the context/target split.

## Evaluating world models

`data/eval/sample5/` holds 5 committed clips (3 through, 2 hidden). It is drawn at
random within each outcome, so the blocked case is always included. Every model lives
in its own folder with its own requirements and Colab notebook, and models never
depend on each other:

| Model | Code | Colab | GPU |
|---|---|---|---|
| Wan 2.1 VACE 1.3B | `models/wan21/` | `notebooks/colab_wan21.ipynb` | T4+ |
| Cosmos-Predict2 2B Video2World | `models/cosmos_predict2/` | `notebooks/colab_cosmos_predict2.ipynb` | A100 |
| LTX-Video 2B distilled (fast) | `models/ltx_video/` | `notebooks/colab_ltx_video.ipynb` | T4+ |
| V-JEPA 2 ViT-L (scored, no pixels) | `models/vjepa2/` | runs locally (MPS/CPU) | any |

Each model writes the same prediction format (see `src/robust_world/eval/io.py`):
- `<clip>.mp4`: the predicted target frames (24)
- `<clip>_input.mp4`: the frames the model was given
- `<clip>.json`: settings

That shared format lets one comparison tool handle any set of models:

```bash
robustworld-eval sample --name sample5 --n 5 --seed 0      # draw a sample
robustworld-eval baseline                                  # hold-last-frame baseline, no GPU needed
robustworld-eval grid --models cosmos_predict2 wan21       # ground truth + models
```

The grid has one column for the ground truth plus one per model, and two rows:
- **Input:** what each source was given. Frames a model didn't see are dimmed.
- **Output:** the ground-truth target or the model's prediction.

The context plays first, then the predictions. The tool writes one video per clip, an
`all.mp4` with every clip stacked, and an `index.html`. Adding a model only adds a column.

## Large files

GitHub rejects any file over 100 MB, so raw videos are not committed (Git LFS
isn't installed yet). `data/interim/` and `data/processed/` are regenerable and
ignored. Copy the raw video to wherever the pipeline runs and rerun it there.

## Layout

| Path                   | Purpose                              | Tracked? |
|------------------------|--------------------------------------|----------|
| `data/raw/videos/`     | Source videos                        | yes (if < 100 MB) |
| `data/interim/`        | Per-video intermediates              | no       |
| `data/processed/`      | Clips and manifests                  | no       |
| `src/robust_world/`    | Pipeline package                     | yes      |
| `configs/scenes/`      | Per-video scene calibration          | yes      |
| `data/eval/`           | Small committed eval samples         | yes      |
| `models/`              | One folder per world model           | yes      |
| `tests/`               | Tests                                | yes      |
| `checkpoints/`         | Model weights                        | no       |
| `outputs/`             | Renders, results                     | no       |
| `notebooks/`           | Exploration                          | yes      |
