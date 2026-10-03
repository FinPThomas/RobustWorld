"""Plumbing test with a tiny randomly initialised LTX-Video (CPU, ~1 min, no real weights).

Builds a miniature Diffusers checkpoint with the same VAE layout as the real 2B model
(0.9.5 architecture, 32x spatial / 8x temporal compression, timestep-conditioned decoder)
and runs predict.py against it. Predictions are noise; only shapes and files are checked.

    python models/ltx_video/test_tiny.py
"""

import json
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))


def build_tiny(path: Path) -> None:
    from diffusers import (AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXConditionPipeline,
                           LTXVideoTransformer3DModel)
    from transformers import AutoTokenizer, T5Config, T5EncoderModel

    torch.manual_seed(0)
    vae = AutoencoderKLLTXVideo(
        in_channels=3, out_channels=3, latent_channels=8,
        block_out_channels=(4, 8, 16, 32, 64), decoder_block_out_channels=(8, 16, 32),
        layers_per_block=(1, 1, 1, 1, 1), decoder_layers_per_block=(1, 1, 1, 1),
        spatio_temporal_scaling=(True, True, True, True), decoder_spatio_temporal_scaling=(True, True, True),
        decoder_inject_noise=(False, False, False, False),
        downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
        upsample_residual=(True, True, True), upsample_factor=(2, 2, 2),
        down_block_types=("LTXVideo095DownBlock3D",) * 4,
        timestep_conditioning=True, patch_size=4, patch_size_t=1, encoder_causal=True, decoder_causal=False,
        spatial_compression_ratio=32, temporal_compression_ratio=8,
    )
    transformer = LTXVideoTransformer3DModel(
        in_channels=8, out_channels=8, patch_size=1, patch_size_t=1, num_attention_heads=2,
        attention_head_dim=8, cross_attention_dim=16, num_layers=1, caption_channels=32)
    text_encoder = T5EncoderModel(T5Config(vocab_size=32128, d_model=32, d_kv=16, d_ff=32, num_layers=1,
                                           num_heads=2))
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
    LTXConditionPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, text_encoder=text_encoder,
                         tokenizer=tokenizer, transformer=transformer).save_pretrained(path)


def main() -> int:
    import predict

    sample = REPO / "data" / "eval" / "sample5"
    clip_id = json.loads((sample / "sample.json").read_text())["clips"][0]["clip_id"]
    with tempfile.TemporaryDirectory() as tmp:
        model, out = Path(tmp) / "tiny-ltx", Path(tmp) / "pred"
        build_tiny(model)
        predict.main(["--sample", str(sample), "--out", str(out), "--checkpoint", "", "--base-repo", str(model),
                      "--size", "64", "--clips", clip_id, "--timesteps", "1000", "500", "0.03"])
        meta = json.loads((out / f"{clip_id}.json").read_text())
        from robust_world.eval.io import read_video
        frames = read_video(out / f"{clip_id}.mp4")
        inputs = read_video(out / f"{clip_id}_input.mp4")
        assert len(frames) == 16 and frames[0].shape == (512, 512, 3), (len(frames), frames[0].shape)
        assert len(inputs) == 9 and meta["input_frame_indices"] == list(range(7, 16))
        assert meta["num_frames"] == 25
    print("ok: tiny LTX-Video produced 16 target frames from the last 9 context frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
