"""Plumbing test with a tiny randomly initialised Wan VACE (CPU, ~1 min, no real weights).

Builds a miniature checkpoint in the real repo layout and runs predict.py against it, so
prompt encoding, freeing the text encoder, VACE extension and the output format are all
exercised. The predictions are noise; only shapes and files are checked.

    python models/wan21/test_tiny.py
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
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline, WanVACETransformer3DModel
    from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

    torch.manual_seed(0)
    transformer = WanVACETransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16, out_channels=16,
        text_dim=32, freq_dim=32, ffn_dim=32, num_layers=2, rope_max_seq_len=32,
        vace_layers=[0], vace_in_channels=96)
    vae = AutoencoderKLWan(base_dim=8, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True])
    text_encoder = UMT5EncoderModel(UMT5Config(vocab_size=32128, d_model=32, d_kv=16, d_ff=32,
                                               num_layers=1, num_heads=2))
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
    scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
    WanVACEPipeline(tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, vae=vae,
                    scheduler=scheduler).save_pretrained(path)


def main() -> int:
    import predict

    sample = REPO / "data" / "eval" / "sample5"
    clip_id = json.loads((sample / "sample.json").read_text())["clips"][0]["clip_id"]
    with tempfile.TemporaryDirectory() as tmp:
        model, out = Path(tmp) / "tiny-wan", Path(tmp) / "pred"
        build_tiny(model)
        predict.main(["--sample", str(sample), "--out", str(out), "--model-id", str(model),
                      "--size", "64", "--steps", "2", "--clips", clip_id])
        meta = json.loads((out / f"{clip_id}.json").read_text())
        from robust_world.eval.io import read_video
        frames = read_video(out / f"{clip_id}.mp4")
        inputs = read_video(out / f"{clip_id}_input.mp4")
        assert len(frames) == 16 and frames[0].shape == (512, 512, 3), (len(frames), frames[0].shape)
        assert len(inputs) == 16 and meta["input_frame_indices"] == list(range(16))
        assert meta["num_frames"] == 33
    print("ok: tiny Wan VACE produced 16 target frames from 16 context frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
