"""Wan 2.1 VACE: predict the target half of each eval clip from its context half.

VACE does video extension natively: the context frames go in as-is, the frames to
generate are grey with a white mask, and the model fills them in. The clip is padded to
the next 4n+1 length Wan needs (24 context + 25 generated = 49); the first 24 generated
frames line up with the ground-truth target.

Memory: every clip shares one text prompt, so the UMT5-XXL text encoder (~11 GB) is
loaded straight onto the GPU, used once and freed before the video model loads. The
1.3B model then fits a Colab T4.

    python models/wan21/predict.py --sample data/eval/sample5 --out outputs/predictions/sample5/wan21
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

MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"   # or Wan-AI/Wan2.1-VACE-14B-diffusers
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "deformed, disfigured, still picture, messy background, camera motion, shaky camera"
)


def pick_dtype(name: str, device: str) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if device == "cuda":
        # bf16 needs Ampere+ (L4, A100); a T4 runs fp16.
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


def encode_prompt(model_id: str, prompt: str, negative: str, dtype, device, cfg: bool):
    """Load only the text encoder, embed the shared prompt once, then free it."""
    from diffusers import WanVACEPipeline
    from transformers import AutoTokenizer, UMT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype,
                                                    low_cpu_mem_usage=True).to(device)
    text_pipe = WanVACEPipeline(tokenizer=tokenizer, text_encoder=text_encoder,
                                transformer=None, vae=None, scheduler=None)
    with torch.no_grad():
        embeds, negative_embeds = text_pipe.encode_prompt(prompt, negative, do_classifier_free_guidance=cfg,
                                                          device=device, dtype=dtype)
    del text_pipe, text_encoder
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return embeds, negative_embeds


def load_video_pipeline(model_id: str, dtype, device: str, flow_shift: float, offload: bool):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline

    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, text_encoder=None, tokenizer=None,
                                           torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    if offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def generated_length(n_context: int, n_target: int) -> int:
    """Smallest 4n+1 total that covers context + target."""
    total = n_context + n_target
    return total + (-(total - 1)) % 4


def predict_clip(pipe, context: list[np.ndarray], n_target: int, embeds, negative_embeds,
                 size: int, steps: int, guidance: float, seed: int, device: str) -> list[np.ndarray]:
    """Return n_target predicted frames (float RGB in [0, 1]) following the context."""
    n_ctx = len(context)
    total = generated_length(n_ctx, n_target)
    frames = [Image.fromarray(f) for f in resize(context, size)]
    frames += [Image.new("RGB", (size, size), (128, 128, 128))] * (total - n_ctx)
    mask = [Image.new("L", (size, size), 0)] * n_ctx + [Image.new("L", (size, size), 255)] * (total - n_ctx)
    out = pipe(
        video=frames, mask=mask,
        prompt_embeds=embeds, negative_prompt_embeds=negative_embeds,
        height=size, width=size, num_frames=total,
        num_inference_steps=steps, guidance_scale=guidance,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="np",
    ).frames[0]
    return list(out[n_ctx:n_ctx + n_target])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--size", type=int, default=512, help="square generation size (multiple of 16)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--flow-shift", type=float, default=3.0, help="3.0 for <=480p-class sizes, 5.0 for 720p")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--offload", action="store_true", help="model CPU offload (less VRAM, slower)")
    p.add_argument("--clips", nargs="*", help="only these clip ids")
    p.add_argument("--prompt", help="override the sample's shared prompt")
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(args.dtype, device)
    sample = load_sample(args.sample)
    prompt = args.prompt or sample["prompt"]
    n_target = sample["target_frames"][1] - sample["target_frames"][0] + 1
    clips = [c for c in sample["clips"] if not args.clips or c["clip_id"] in args.clips]
    print(f"Wan 2.1 VACE ({args.model_id}) on {device} {dtype}; {len(clips)} clips")

    embeds, negative_embeds = encode_prompt(args.model_id, prompt, NEGATIVE_PROMPT, dtype, device,
                                            cfg=args.guidance > 1)
    pipe = load_video_pipeline(args.model_id, dtype, device, args.flow_shift, args.offload)

    for clip in clips:
        t0 = time.time()
        context, _ = read_clip(sample, clip["clip_id"])
        pred = predict_clip(pipe, context, n_target, embeds, negative_embeds,
                            args.size, args.steps, args.guidance, args.seed, device)
        seconds = round(time.time() - t0, 1)
        write_prediction(args.out, clip["clip_id"], pred, context, list(range(len(context))), {
            "model": "wan21", "label": "Wan 2.1 VACE " + ("14B" if "14B" in args.model_id else "1.3B"),
            "model_id": args.model_id, "prompt": prompt, "size": args.size, "steps": args.steps,
            "guidance": args.guidance, "flow_shift": args.flow_shift, "seed": args.seed,
            "num_frames": generated_length(len(context), n_target), "dtype": str(dtype),
            "seconds": seconds,
        })
        print(f"  {clip['clip_id']}: {seconds}s")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
