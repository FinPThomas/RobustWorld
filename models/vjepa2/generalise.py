"""Does what post-training learnt carry over? The ball rolled from the other side, and a new ball.

Held-out clips are never trained on or used to pick anything:
    other side   clips whose pass enters from the side opposite the training direction
                 (passes.json "side_in"); the training direction is the most common side among the
                 clips of the training video(s).
    new ball     every clip of a video listed in configs/heldout.json {"videos": ["<video stem>", ...]}
                 (or --holdout-videos).
Everything else is "seen" and is what the plan's cross-validated stages train and score on.

`split` writes the seen manifest (data/processed/clips/manifest_seen.jsonl) and split.json, and
exits with code 3 when there are no held-out clips yet (the plan then waits for the recordings).
`score` reads every held-out clip with the frozen evaluation decoder, fitted (as always) on real
context-half features of seen clips only, labelled with same-frame ball positions
(eval_decoder.examples_from). It compares the pretrained predictor with one post-trained on all seen
clips (posttrain.py train --all --manifest manifest_seen.jsonl). Far/near cells follow each clip's
direction: "far" is the side opposite the one the ball came from. Nothing here is fitted on
outcomes, target halves or predictions (CLAUDE.md).

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
MANIFEST = CLIPS / "manifest.jsonl"
SEEN = CLIPS / "manifest_seen.jsonl"
HELDOUT_CFG = REPO / "configs" / "heldout.json"
OUT = REPO / "outputs" / "vjepa2" / "plan" / "generalise"
WAITING = 3          # exit code: no held-out clips yet


def stem(clip: dict) -> str:
    return Path(clip["source_video"]).stem


@lru_cache(maxsize=None)
def tracking(video: str):
    return load_tracking(video)


def side_in(clip: dict) -> str | None:
    return tracking(stem(clip))[1][clip["pass_id"]].get("side_in")


def split(clips: list[dict], holdout_videos: list[str]) -> tuple[list[dict], dict[str, list[dict]], str]:
    """-> (seen clips, held-out clips by group, training direction)."""
    held = {v for v in holdout_videos}
    sides = Counter(side_in(c) for c in clips if stem(c) not in held)
    train_side = sides.most_common(1)[0][0] if sides else None
    seen, groups = [], {}
    for c in clips:
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


def holdout_videos(arg: list[str] | None) -> list[str]:
    if arg is not None:
        return arg
    return json.loads(HELDOUT_CFG.read_text()).get("videos", []) if HELDOUT_CFG.exists() else []


def cmd_split(args) -> int:
    clips = included(args.manifest)
    seen, groups, train_side = split(clips, holdout_videos(args.holdout_videos))
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
        print("no held-out clips yet: record the ball from the other side, or list a new-ball video in "
              "configs/heldout.json, then rerun")
        return WAITING
    return 0


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
    seen, groups, _ = split(clips, holdout_videos(args.holdout_videos))
    if not groups:
        print("no held-out clips yet")
        return WAITING
    device = pick_device()
    model = VJEPA2Model.from_pretrained(args.model_id or MODEL_ID)
    tub, grid = model.config.tubelet_size, SIZE // model.config.patch_size
    model.encoder = None
    model.to(device).eval()
    base = {k: v.detach().clone() for k, v in model.predictor.state_dict().items()}
    cache = cache_dir(args.model_id or MODEL_ID)

    def item(c):
        enc = load_cached(cache, c)
        if enc is None:
            raise SystemExit(f"{c['clip_id']} is not encoded; run posttrain.py encode on the full manifest")
        track, passes, _ = tracking(stem(c))
        n_ctx = c["context_frames"][1] + 1
        lab, centres = ball_labels(source_indices(c, passes, c["n_frames"], n_ctx), track, grid, tub)
        return enc, lab, centres

    # The frozen decoder: real context halves of seen clips, same-frame labels, nothing else.
    decoder = eval_decoder.fit([eval_decoder.examples_from(*item(c)[:2]) for c in seen], seed=0)
    meta = load_predictor(model, args.run / "all.pt")
    trained = set(meta["train_clip_ids"])
    after_state = {k: v.detach().clone() for k, v in model.predictor.state_dict().items()}

    results, per_clip = {}, []
    for g, cs in groups.items():
        if trained & {c["clip_id"] for c in cs}:
            raise SystemExit(f"{args.run}/all.pt was trained on held-out clips ({g})")
        rows = {k: [] for k in ("real", "before", "after")}
        maps = {k: [] for k in rows}
        labs, centres_all = [], []
        for c in cs:
            enc, lab, centres = item(c)
            scene = tracking(stem(c))[2]
            far, near = sides_for(c, scene, grid)
            cs_ = enc["context"].shape[0]
            ctx = enc["context"][None].float().to(device)
            steps = enc["real"].shape[0] - cs_
            with torch.no_grad():
                model.predictor.load_state_dict(base)
                before = imagine(model.predictor, ctx, steps)[0].float().cpu()
                model.predictor.load_state_dict(after_state)
                after = imagine_mode(model.predictor, ctx, steps, meta)[0].float().cpu()
            out = {"real": decoder(enc["real"][cs_:]).numpy(), "before": decoder(before).numpy(),
                   "after": decoder(after).numpy()}
            labs.append(lab[cs_:])
            centres_all.append(centres[cs_:])
            rec = {"clip_id": c["clip_id"], "group": g, "outcome": c["outcome"]}
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
        print(f"{g}: {len(cs)} clips; through-vs-blocked AUROC before {r['before']['outcome_auroc']} -> after "
              f"{r['after']['outcome_auroc']} (real frames {r['real']['outcome_auroc']})")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps({"run": str(args.run), "groups": results}, indent=1))
    (args.out / "per_clip.json").write_text(json.dumps(per_clip, indent=1))
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
