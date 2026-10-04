"""NVIDIA Cosmos-Predict2 Video2World: predict the target half of each eval clip.

Cosmos-Predict2 conditions on the last 5 frames of the input video (its trained multi-frame
mode) and rolls the world forward. It is given context frames 19-23 and generates
--num-frames in total (93 = its native 5.8 s at 16 fps); generated frames 5..28 line up with
the ground-truth target frames 24..47.

The checkpoint is 720p. Square clips are upscaled to 960x960 (its 1:1 720p shape), predicted,
and downscaled back to 512.

Licence: the NVIDIA Open Model License forbids disabling the Cosmos guardrail. The pipeline
skips its own text check when given precomputed prompt embeddings, so this script runs that
check on the prompt itself, and the pipeline still checks every generated video.

Memory: the T5-11B text encoder runs once for the shared prompt and is freed before the
video model loads. Needs ~40 GB of GPU memory (Colab A100) at the default settings.

    python models/cosmos_predict2/predict.py --sample data/eval/sample5 --out outputs/predictions/sample5/cosmos_predict2
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

MODEL_ID = "nvidia/Cosmos-Predict2-2B-Video2World"   # or nvidia/Cosmos-Predict2-14B-Video2World
NUM_CONDITIONAL_FRAMES = 5
NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
    "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)


def load_safety_checker():
    from cosmos_guardrail import CosmosSafetyChecker
    return CosmosSafetyChecker()


def check_prompt(safety_checker, prompt: str, device: str) -> None:
    """The text half of the Cosmos guardrail (the pipeline skips it when given embeddings)."""
    safety_checker.to(device)
    try:
        if not safety_checker.check_text_safety(prompt):
            raise ValueError(f"Cosmos guardrail rejected the prompt: {prompt!r}")
    finally:
        safety_checker.to("cpu")


def encode_prompt(model_id: str, prompt: str, negative: str, dtype, device, cfg: bool, safety_checker):
    """Load only the T5 text encoder, embed the shared prompt once, then free it."""
    from diffusers import Cosmos2VideoToWorldPipeline
    from transformers import T5EncoderModel, T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype,
                                                  low_cpu_mem_usage=True).to(device)
    text_pipe = Cosmos2VideoToWorldPipeline(tokenizer=tokenizer, text_encoder=text_encoder, transformer=None,
                                            vae=None, scheduler=None, safety_checker=safety_checker)
    with torch.no_grad():
        embeds, negative_embeds = text_pipe.encode_prompt(prompt, negative, do_classifier_free_guidance=cfg,
                                                          device=device, dtype=dtype)
    del text_pipe, text_encoder
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return embeds, negative_embeds


def load_video_pipeline(model_id: str, dtype, device: str, offload: bool, safety_checker):
    from diffusers import Cosmos2VideoToWorldPipeline

    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(model_id, text_encoder=None, tokenizer=None,
                                                       safety_checker=safety_checker, torch_dtype=dtype)
    if offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        # Not pipe.to(): that would also park the large guardrail models on the GPU. The
        # pipeline moves the guardrail there itself only while it checks each video.
        pipe.transformer.to(device)
        pipe.vae.to(device)
    return pipe


def predict_clip(pipe, context: list[np.ndarray], n_target: int, embeds, negative_embeds, size: int,
                 num_frames: int, steps: int, guidance: float, seed: int) -> tuple[list[np.ndarray], list[int]]:
    """Return (n_target predicted frames as float RGB in [0, 1], clip indices of the frames given)."""
    n_ctx = len(context)
    given = list(range(n_ctx - NUM_CONDITIONAL_FRAMES, n_ctx))
    if num_frames < NUM_CONDITIONAL_FRAMES + n_target or (num_frames - 1) % 4:
        raise ValueError(f"--num-frames must be 4n+1 and >= {NUM_CONDITIONAL_FRAMES + n_target}")
    video = [Image.fromarray(f) for f in resize([context[i] for i in given], size)]
    out = pipe(
        video=video,
        prompt_embeds=embeds, negative_prompt_embeds=negative_embeds,
        height=size, width=size, num_frames=num_frames, fps=16,
        num_inference_steps=steps, guidance_scale=guidance,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="np",
    ).frames[0]
    pred = out[NUM_CONDITIONAL_FRAMES:NUM_CONDITIONAL_FRAMES + n_target]
    return list(pred), given


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--size", type=int, default=960, help="square generation size; 960 = 1:1 at 720p")
    p.add_argument("--num-frames", type=int, default=93,
                   help="total frames generated incl. the 5 conditioning ones (4n+1, >= 21). "
                        "93 is native; 29 or 21 is much faster but outside the trained length")
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offload", action="store_true", help="model CPU offload (less VRAM, slower)")
    p.add_argument("--clips", nargs="*", help="only these clip ids")
    p.add_argument("--prompt", help="override the sample's shared prompt")
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    sample = load_sample(args.sample)
    prompt = args.prompt or sample["prompt"]
    n_target = sample["target_frames"][1] - sample["target_frames"][0] + 1
    clips = [c for c in sample["clips"] if not args.clips or c["clip_id"] in args.clips]
    print(f"Cosmos-Predict2 Video2World ({args.model_id}) on {device} {dtype}; {len(clips)} clips")

    safety_checker = load_safety_checker()
    check_prompt(safety_checker, prompt, device)
    embeds, negative_embeds = encode_prompt(args.model_id, prompt, NEGATIVE_PROMPT, dtype, device,
                                            args.guidance > 1, safety_checker)
    pipe = load_video_pipeline(args.model_id, dtype, device, args.offload, safety_checker)

    for clip in clips:
        t0 = time.time()
        context, _ = read_clip(sample, clip["clip_id"])
        pred, given = predict_clip(pipe, context, n_target, embeds, negative_embeds, args.size,
                                   args.num_frames, args.steps, args.guidance, args.seed)
        seconds = round(time.time() - t0, 1)
        write_prediction(args.out, clip["clip_id"], pred, [context[i] for i in given], given, {
            "model": "cosmos_predict2",
            "label": "Cosmos-Predict2 " + ("14B" if "14B" in args.model_id else "2B"),
            "model_id": args.model_id, "prompt": prompt, "size": args.size, "num_frames": args.num_frames,
            "steps": args.steps, "guidance": args.guidance, "seed": args.seed, "seconds": seconds,
        })
        print(f"  {clip['clip_id']}: {seconds}s")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
