"""Does what post-training learnt carry over? The ball rolled from the other side, and a new ball.

Held-out clips are never trained on or used to pick anything:
    other side   clips whose pass enters from the side opposite the training direction
                 (passes.json "side_in"); the training direction is the most common side among the
                 clips of the training video(s).
    new ball     every clip of a video listed in configs/heldout.json {"videos": ["<video stem>", ...]}
                 (or --holdout-videos).
Everything else is "seen" and is what the plan's cross-validated stages train and score on.

`split` writes the seen manifest (outputs/vjepa2/plan/manifest_seen.jsonl) and split.json, and
exits with code 3 when there are no held-out clips yet (the plan then waits for the recordings).
`score` reads every held-out clip with the frozen evaluation decoder, fitted (as always) on real
context-half features of seen clips only, labelled with same-frame ball positions
(eval_decoder.examples_from). It compares the pretrained predictor with one post-trained on all seen
clips (posttrain.py train --all --manifest manifest_seen.jsonl). Far/near cells follow each clip's
direction: "far" is the side opposite the one the ball came from. Nothing here is fitted on
outcomes, target halves or predictions (CLAUDE.md).

Two checks of what the post-trained predictor learnt (a rule tied to the training direction, or where
the blockade is):
    mirror  each held-out clip is flipped left to right (so the ball comes in from the training side),
            encoded, imagined and decoded, and the maps are flipped back before scoring ("mirror_*").
            Good scores here but not on the clips as filmed: it learnt a rule for one direction.
    height  the height at which the ball reaches the plank (last tracked centre of the context half).
            "height_rule" is the blockade range that best separates outcomes on the seen clips (the
            TAPNext baseline's fit_blockade, a reference fitted on outcomes, never applied to a model's
            output): its accuracy on the held-out clips says whether the blockade blocks at the same
            heights from the other side, and each phase's auroc_vs_height_rule says whether the model's
            "gets through" score follows the heights.

    python models/vjepa2/generalise.py split
    python models/vjepa2/generalise.py score --run checkpoints/vjepa2/plan/<variant>-all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from robust_world.eval.ball import (OUTCOMES, ball_labels, ball_track_metrics, far_cells,  # noqa: E402
                                    load_tracking, near_cells, one_ball_readouts, outcome_metrics,
                                    roc_auc_score_safe, source_indices)

CLIPS = REPO / "data" / "processed" / "clips"
# Training clips: the right-entry segment (whole.mp4 pipeline); older single-video data has manifest.jsonl.
MANIFEST = CLIPS / "manifest_right.jsonl" if (CLIPS / "manifest_right.jsonl").exists() else CLIPS / "manifest.jsonl"
SEEN = REPO / "outputs" / "vjepa2" / "plan" / "manifest_seen.jsonl"   # on Drive in Colab, so it survives restarts
HELDOUT_CFG = REPO / "configs" / "heldout.json"
OUT = REPO / "outputs" / "vjepa2" / "plan" / "generalise"
WAITING = 3          # exit code: no held-out clips yet
PHASES = ("real", "before", "after", "mirror_real", "mirror_before", "mirror_after")


def stem(clip: dict) -> str:
    """The video's name: source_video is data/interim/<name>/<name>_512.mp4 (as in pack_clips.py)."""
    return Path(clip["source_video"]).parent.name


@lru_cache(maxsize=None)
def tracking(video: str):
    return load_tracking(video)


def side_in(clip: dict) -> str | None:
    if clip.get("side_in"):
        return clip["side_in"]
    return tracking(stem(clip))[1][clip["pass_id"]].get("side_in")


def split(clips: list[dict], holdout_videos: list[str], others: list[dict] = ()) -> tuple[list[dict], dict[str, list[dict]], str]:
    """clips: the training manifest's; others: other segments' (e.g. manifest_left.jsonl), always held out.
    -> (seen clips, held-out clips by group, training direction)."""
    held = {v for v in holdout_videos}
    sides = Counter(side_in(c) for c in clips if stem(c) not in held)
    train_side = sides.most_common(1)[0][0] if sides else None
    seen, groups = [], {}
    for c in [*clips, *others]:
        if stem(c) in held:
            groups.setdefault(f"new ball ({stem(c)})", []).append(c)
        elif side_in(c) != train_side:
            groups.setdefault(f"other side (from {side_in(c)})", []).append(c)
        else:
            seen.append(c)
    return seen, groups, train_side


def included(manifest: Path) -> list[dict]:
    clips = [json.loads(line) for line in manifest.open()]
    return [c for c in clips if c.get("include") and c["outcome"] in OUTCOMES]


def other_segments(manifest: Path) -> list[dict]:
    """Included clips of every other segment manifest next to the training one (manifest_<segment>.jsonl)."""
    seen = {c["clip_id"] for c in included(manifest)}
    out = []
    for m in sorted(manifest.parent.glob("manifest_*.jsonl")):
        if m.resolve() != manifest.resolve():
            out += [c for c in included(m) if c["clip_id"] not in seen]
    return out


def holdout_videos(arg: list[str] | None) -> list[str]:
    if arg is not None:
        return arg
    return json.loads(HELDOUT_CFG.read_text()).get("videos", []) if HELDOUT_CFG.exists() else []


def cmd_split(args) -> int:
    clips = included(args.manifest)
    seen, groups, train_side = split(clips, holdout_videos(args.holdout_videos), other_segments(args.manifest))
    SEEN.parent.mkdir(parents=True, exist_ok=True)
    SEEN.write_text("".join(json.dumps(c) + "\n" for c in seen))
    args.out.mkdir(parents=True, exist_ok=True)
    info = {"training_direction": train_side, "n_seen": len(seen),
            "held_out": {g: {"n": len(cs), **{o: sum(c["outcome"] == o for c in cs) for o in OUTCOMES}}
                         for g, cs in groups.items()}}
    (args.out / "split.json").write_text(json.dumps(info, indent=1))
    print(f"seen: {len(seen)} clips (ball from {train_side}) -> {SEEN}")
    for g, v in info["held_out"].items():
        print(f"held out: {g}: {v}")
    if not groups:
        print("no held-out clips yet: pack the left segment (scripts/pack_clips.py --manifest "
              "data/processed/clips/manifest_left.jsonl --out outputs/robustworld_clips_left.zip) and put it on Drive, "
              "or list a new-ball video in configs/heldout.json, then rerun")
        return WAITING
    return 0


def mirror_cache(cache: Path) -> Path:
    return cache / "mirror"


def encode_mirrored(model, clips: list[dict], cache: Path, model_id: str, device) -> None:
    """Encode each clip flipped left to right (once; kept next to the normal encodings)."""
    import ball_probe_cv
    from ball_probe import encode
    from posttrain import load_cached, save_atomic
    from robust_world.eval.io import read_video

    out = mirror_cache(cache)
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in clips if load_cached(out, c) is None]
    if not todo:
        return
    mean, std = ball_probe_cv.normalisation(model_id)
    for i, c in enumerate(todo):
        frames = [np.ascontiguousarray(f[:, ::-1]) for f in read_video(REPO / c["path"])]
        enc = encode(model, frames, c["context_frames"][1] + 1, mean, std, device, imagine=True)
        save_atomic(enc, out / f"{c['clip_id']}.pt")
        print(f"\r  mirrored {i + 1}/{len(todo)} clips", end="", flush=True)
    print()


def entry_height(centres: list, n_ctx_steps: int) -> float | None:
    """y of the last tracked ball centre in the context half: about where the ball reaches the plank."""
    ys = [c[1] for c in centres[:n_ctx_steps] if c is not None]
    return float(ys[-1]) if ys else None


def fit_blockade(rows: list[dict]):
    """The TAPNext baseline's blockade range (fitted on outcomes: a reference, see the module docstring)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tapnext_predict", REPO / "models" / "tapnext_rule" / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fit_blockade(rows)


def height_report(per_clip: list[dict], blockade, phases) -> dict:
    rows = [r for r in per_clip if r["entry_y"] is not None]
    if blockade is None or not rows:
        return {"range": None}
    rule_through = np.array([not (blockade[0] <= r["entry_y"] <= blockade[1]) for r in rows], int)
    real_through = np.array([r["outcome"] == "through" for r in rows], int)
    return {"range": [round(blockade[0], 1), round(blockade[1], 1)], "n_with_height": len(rows),
            "accuracy_on_real_outcomes": round(float(np.mean(rule_through == real_through)), 3),
            "auroc_vs_height_rule": {k: roc_auc_score_safe(rule_through, np.array([max(r[f"{k}_far"]) for r in rows]))
                                     for k in phases}}


def height_figure(per_clip: list[dict], blockade, phases, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"through": "tab:green", "hidden": "tab:red", "bounce": "tab:orange"}
    fig, axes = plt.subplots(1, len(phases), figsize=(3.2 * len(phases), 3.2), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), phases):
        for o, col in colours.items():
            rs = [r for r in per_clip if r["outcome"] == o and r["entry_y"] is not None]
            ax.scatter([r["entry_y"] for r in rs], [max(r[f"{k}_far"]) for r in rs], s=10, c=col, label=o)
        if blockade is not None:
            ax.axvspan(*blockade, color="grey", alpha=0.15)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("height reaching the plank (px)", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("ball beyond the plank (peak)", fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Held-out clips: grey = blockade range fitted on the seen clips", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def sides_for(clip: dict, scene: dict, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """(far, near) cells for this clip's direction."""
    far, near = far_cells(scene, grid), near_cells(scene, grid)    # far_cells: beyond the plank for a ball from R
    return (far, near) if side_in(clip) == "R" else (near, far)


def cmd_score(args) -> int:
    import torch
    from transformers import VJEPA2Model

    import eval_decoder
    from posttrain import cache_dir, imagine, imagine_mode, load_cached, load_predictor
    from run import MODEL_ID, SIZE, pick_device

    clips = included(args.manifest)
    seen, groups, _ = split(clips, holdout_videos(args.holdout_videos), other_segments(args.manifest))
    if not groups:
        print("no held-out clips yet")
        return WAITING
    device = pick_device()
    model = VJEPA2Model.from_pretrained(args.model_id or MODEL_ID)
    tub, grid = model.config.tubelet_size, SIZE // model.config.patch_size
    model.to(device).eval()
    cache = cache_dir(args.model_id or MODEL_ID)
    print("== encoding the held-out clips flipped left to right (mirror test)", flush=True)
    encode_mirrored(model, [c for cs in groups.values() for c in cs], cache, args.model_id or MODEL_ID, device)
    model.encoder = None
    if device == "cuda" or getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    base = {k: v.detach().clone() for k, v in model.predictor.state_dict().items()}

    def item(c):
        enc = load_cached(cache, c)
        if enc is None:
            raise SystemExit(f"{c['clip_id']} is not encoded; run posttrain.py encode on the full manifest")
        track, passes, _ = tracking(stem(c))
        n_ctx = c["context_frames"][1] + 1
        lab, centres = ball_labels(source_indices(c, passes, c["n_frames"], n_ctx), track, grid, tub)
        return enc, lab, centres

    # The frozen decoder: real context halves of seen clips, same-frame labels, nothing else.
    examples, seen_heights = [], []
    for c in seen:
        enc, lab, centres = item(c)
        examples.append(eval_decoder.examples_from(enc, lab))
        seen_heights.append({"crossing_y": entry_height(centres, enc["context"].shape[0]), "outcome": c["outcome"]})
    decoder = eval_decoder.fit(examples, seed=0)
    del examples
    blockade = fit_blockade(seen_heights)          # reference only: never applied to a model's output
    meta = load_predictor(model, args.run / "all.pt")
    trained = set(meta["train_clip_ids"])
    after_state = {k: v.detach().clone() for k, v in model.predictor.state_dict().items()}

    results, per_clip = {}, []
    for g, cs in groups.items():
        if trained & {c["clip_id"] for c in cs}:
            raise SystemExit(f"{args.run}/all.pt was trained on held-out clips ({g})")
        rows = {k: [] for k in PHASES}
        maps = {k: [] for k in rows}
        labs, centres_all = [], []
        for c in cs:
            enc, lab, centres = item(c)
            scene = tracking(stem(c))[2]
            far, near = sides_for(c, scene, grid)
            cs_ = enc["context"].shape[0]
            ctx = enc["context"][None].float().to(device)
            steps = enc["real"].shape[0] - cs_
            menc = load_cached(mirror_cache(cache), c)
            mctx = menc["context"][None].float().to(device)
            with torch.no_grad():
                model.predictor.load_state_dict(base)
                before = imagine(model.predictor, ctx, steps)[0].float().cpu()
                mbefore = imagine(model.predictor, mctx, steps)[0].float().cpu()
                model.predictor.load_state_dict(after_state)
                after = imagine_mode(model.predictor, ctx, steps, meta)[0].float().cpu()
                mafter = imagine_mode(model.predictor, mctx, steps, meta)[0].float().cpu()
            unflip = lambda m: np.ascontiguousarray(m[..., ::-1])  # noqa: E731  back to the clip as filmed
            out = {"real": decoder(enc["real"][cs_:]).numpy(), "before": decoder(before).numpy(),
                   "after": decoder(after).numpy(), "mirror_real": unflip(decoder(menc["real"][cs_:]).numpy()),
                   "mirror_before": unflip(decoder(mbefore).numpy()), "mirror_after": unflip(decoder(mafter).numpy())}
            labs.append(lab[cs_:])
            centres_all.append(centres[cs_:])
            rec = {"clip_id": c["clip_id"], "group": g, "outcome": c["outcome"],
                   "entry_y": entry_height(centres, cs_)}
            for k, m in out.items():
                ro = one_ball_readouts(m, far, near)
                rows[k].append({"outcome": c["outcome"], **ro})
                maps[k].append(m)
                rec[f"{k}_far"], rec[f"{k}_near"] = ro["far"], ro["near"]
            per_clip.append(rec)
        fl = np.concatenate([x.reshape(-1) for x in labs])
        results[g] = {"n": len(cs), **{f"n_{o}": sum(c["outcome"] == o for c in cs) for o in OUTCOMES},
                      **{k: {**outcome_metrics(rows[k]),
                             "cell_auroc": round(float(roc_auc_score_safe(fl, np.concatenate(
                                 [m.reshape(-1) for m in maps[k]]))), 3),
                             "argmax_hit_rate": ball_track_metrics(maps[k], centres_all, confident=0.0)["hit_rate"]}
                         for k in rows}}
        r = results[g]
        r["height"] = height_report([p for p in per_clip if p["group"] == g], blockade, PHASES)
        print(f"{g}: {len(cs)} clips; through-vs-blocked AUROC before {r['before']['outcome_auroc']} -> after "
              f"{r['after']['outcome_auroc']} (real frames {r['real']['outcome_auroc']}); mirrored: before "
              f"{r['mirror_before']['outcome_auroc']} -> after {r['mirror_after']['outcome_auroc']} (real frames "
              f"{r['mirror_real']['outcome_auroc']}); height rule {r['height']}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps({"run": str(args.run), "groups": results}, indent=1))
    (args.out / "per_clip.json").write_text(json.dumps(per_clip, indent=1))
    height_figure(per_clip, blockade, ("real", "before", "after", "mirror_after"), args.out / "height.png")
    print(f"-> {args.out}/ (metrics.json, per_clip.json)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["split", "score"])
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--holdout-videos", nargs="*", default=None, help="video stems held out as the new-ball test")
    p.add_argument("--run", type=Path, help="score: posttrain.py --all run folder (with all.pt)")
    p.add_argument("--model-id", default=None)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    return cmd_split(args) if args.stage == "split" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
