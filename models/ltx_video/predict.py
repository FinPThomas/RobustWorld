"""LTX-Video 2B distilled (0.9.8): predict the target half of each eval clip, fast.

LTX's VAE compresses time 8x, so conditioning videos must be 8k+1 frames long, and the
pipeline trims longer ones from the *start*. The model is therefore given the last 17
context frames (clip frames 7-23) and generates 41 frames (clip frames 7-47);
generated frames 17-40 line up with the ground-truth target frames 24-47.

The distilled model needs no classifier-free guidance and 8 fixed timesteps, so a clip
takes seconds rather than minutes. Its weights ship as one file holding the transformer
and VAE (the 0.9.5 architecture); the T5 text encoder and scheduler come from a Diffusers
repo. The text encoder (~9.5 GB) runs once for the shared prompt and is freed first.

    python models/ltx_video/predict.py --sample data/eval/sample5 --out outputs/predictions/sample5/ltx_video
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robust_world.eval.io import load_sample, read_clip, resize, write_prediction  # noqa: E402

CHECKPOINT = "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.8-distilled.safetensors"
CHECKPOINT_CONFIG = "Lightricks/LTX-Video-0.9.5"       # Diffusers config matching the 2B single file
BASE_REPO = "Lightricks/LTX-Video-0.9.7-distilled"     # T5 text encoder, tokenizer, distilled scheduler
DISTILLED_TIMESTEPS = [1000, 993, 987, 981, 975, 909, 725, 0.03]
TEMPORAL_COMPRESSION = 8
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted, camera motion"


def pick_dtype(name: str, device: str) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def encode_prompt(base_repo: str, prompt: str, dtype, device, offload_dir: Path | None = None):
    """Load only the T5 text encoder, embed the shared prompt once, then free it.

    With offload_dir (machines with less RAM than the ~9.5 GB encoder), weights stay on disk and
    stream through the CPU layer by layer; slow, but it only runs once.
    """
    from diffusers import LTXConditionPipeline
    from transformers import T5EncoderModel, T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained(base_repo, subfolder="tokenizer")
    if offload_dir:
        text_encoder = T5EncoderModel.from_pretrained(
            base_repo, subfolder="text_encoder", torch_dtype=dtype, device_map="auto",
            max_memory={"cpu": "2GiB"}, offload_folder=str(offload_dir))
        device_for_text = "cpu"
    else:
        text_encoder = T5EncoderModel.from_pretrained(base_repo, subfolder="text_encoder", torch_dtype=dtype,
                                                      low_cpu_mem_usage=True).to(device)
        device_for_text = device
    text_pipe = LTXConditionPipeline(tokenizer=tokenizer, text_encoder=text_encoder, transformer=None,
                                     vae=None, scheduler=None)
    with torch.no_grad():
        embeds, mask, _, _ = text_pipe.encode_prompt(prompt, do_classifier_free_guidance=False,
                                                     device=device_for_text, dtype=dtype)
    del text_pipe, text_encoder
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return embeds.to(device), mask.to(device)


def load_video_pipeline(checkpoint: str | None, base_repo: str, dtype, device: str):
    from diffusers import AutoencoderKLLTXVideo, LTXConditionPipeline, LTXVideoTransformer3DModel

    parts = {}
    if checkpoint:
        parts["transformer"] = LTXVideoTransformer3DModel.from_single_file(
            checkpoint, config=CHECKPOINT_CONFIG, subfolder="transformer", torch_dtype=dtype)
        parts["vae"] = AutoencoderKLLTXVideo.from_single_file(
            checkpoint, config=CHECKPOINT_CONFIG, subfolder="vae", torch_dtype=dtype)
    pipe = LTXConditionPipeline.from_pretrained(base_repo, text_encoder=None, tokenizer=None,
                                                torch_dtype=dtype, **parts)
    pipe.to(device)
    return pipe


def condition_length(n_context: int) -> int:
    """Longest 8k+1 run of context frames."""
    return (n_context - 1) // TEMPORAL_COMPRESSION * TEMPORAL_COMPRESSION + 1


def generated_length(n_cond: int, n_target: int) -> int:
    """Smallest 8k+1 total that covers the conditioning frames + target."""
    total = n_cond + n_target
    return total + (-(total - 1)) % TEMPORAL_COMPRESSION


def predict_clip(pipe, context: list[np.ndarray], n_target: int, embeds, mask, size: int,
                 frame_rate: float, seed: int, timesteps: list[float]) -> tuple[list[np.ndarray], list[int]]:
    """Return (n_target predicted frames as float RGB in [0, 1], clip indices of the frames given)."""
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

    n_ctx = len(context)
    n_cond = condition_length(n_ctx)
    given = list(range(n_ctx - n_cond, n_ctx))
    total = generated_length(n_cond, n_target)
    video = [Image.fromarray(f) for f in resize([context[i] for i in given], size)]
    out = pipe(
        conditions=[LTXVideoCondition(video=video, frame_index=0, strength=1.0)],
        prompt_embeds=embeds, prompt_attention_mask=mask,
        height=size, width=size, num_frames=total, frame_rate=frame_rate,
        timesteps=timesteps, guidance_scale=1.0,
        decode_timestep=0.05, decode_noise_scale=0.025, image_cond_noise_scale=0.0,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="np",
    ).frames[0]
    return list(out[n_cond:n_cond + n_target]), given


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", default=CHECKPOINT,
                   help="single-file transformer+VAE; pass '' to load them from --base-repo instead")
    p.add_argument("--base-repo", default=BASE_REPO)
    p.add_argument("--size", type=int, default=512, help="square generation size (multiple of 32)")
    p.add_argument("--frame-rate", type=float, default=16.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--clips", nargs="*", help="only these clip ids")
    p.add_argument("--prompt", help="override the sample's shared prompt")
    p.add_argument("--timesteps", type=float, nargs="+", default=DISTILLED_TIMESTEPS)
    p.add_argument("--text-encoder-offload", type=Path,
                   help="stream the T5 encoder from this disk folder (low-RAM machines)")
    args = p.parse_args(argv)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = pick_dtype(args.dtype, device)
    sample = load_sample(args.sample)
    prompt = args.prompt or sample["prompt"]
    n_target = sample["target_frames"][1] - sample["target_frames"][0] + 1
    clips = [c for c in sample["clips"] if not args.clips or c["clip_id"] in args.clips]
    print(f"LTX-Video 2B distilled on {device} {dtype}; {len(clips)} clips")

    embeds, mask = encode_prompt(args.base_repo, prompt, dtype, device, args.text_encoder_offload)
    pipe = load_video_pipeline(args.checkpoint or None, args.base_repo, dtype, device)

    for clip in clips:
        t0 = time.time()
        context, _ = read_clip(sample, clip["clip_id"])
        pred, given = predict_clip(pipe, context, n_target, embeds, mask, args.size, args.frame_rate,
                                   args.seed, args.timesteps)
        seconds = round(time.time() - t0, 1)
        write_prediction(args.out, clip["clip_id"], pred, [context[i] for i in given], given, {
            "model": "ltx_video", "label": "LTX-Video 2B distilled",
            "checkpoint": args.checkpoint, "base_repo": args.base_repo, "prompt": prompt,
            "size": args.size, "frame_rate": args.frame_rate, "timesteps": args.timesteps,
            "num_frames": generated_length(len(given), n_target), "seed": args.seed,
            "dtype": str(dtype), "seconds": seconds,
        })
        print(f"  {clip['clip_id']}: {seconds}s")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
