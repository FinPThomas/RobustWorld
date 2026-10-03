"""Plumbing test with a tiny randomly initialised Cosmos-Predict2 (CPU, ~1 min, no real weights).

Builds a miniature checkpoint in the real repo layout and runs predict.py against it. The
real guardrail pulls several large models, so this test swaps in a pass-through stub; that
is only acceptable because no NVIDIA weights are involved. predict.py itself always loads
the real cosmos_guardrail.

    python models/cosmos_predict2/test_tiny.py
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


class StubGuardrail(torch.nn.Module):
    dtype = torch.float32
    device = torch.device("cpu")

    def check_text_safety(self, prompt):
        return True

    def check_video_safety(self, video):
        return video


def build_tiny(path: Path) -> None:
    from diffusers import AutoencoderKLWan, CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, T5Config, T5EncoderModel

    torch.manual_seed(0)
    CosmosTransformer3DModel(
        in_channels=17, out_channels=16, num_attention_heads=2, attention_head_dim=16, num_layers=2,
        mlp_ratio=2, text_embed_dim=32, adaln_lora_dim=4, max_size=(32, 32, 32), patch_size=(1, 2, 2),
        rope_scale=(2.0, 1.0, 1.0), concat_padding_mask=True, extra_pos_embed_type="learnable",
    ).save_pretrained(path / "transformer")
    AutoencoderKLWan(base_dim=8, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                     temperal_downsample=[False, True, True]).save_pretrained(path / "vae")
    T5EncoderModel(T5Config(vocab_size=32128, d_model=32, d_kv=16, d_ff=32, num_layers=1,
                            num_heads=2)).save_pretrained(path / "text_encoder")
    AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5").save_pretrained(path / "tokenizer")
    FlowMatchEulerDiscreteScheduler(use_karras_sigmas=True, shift=1.0).save_pretrained(path / "scheduler")
    (path / "model_index.json").write_text(json.dumps({
        "_class_name": "Cosmos2VideoToWorldPipeline",
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", "CosmosTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "safety_checker": [None, None],
    }))


def main() -> int:
    import predict

    predict.load_safety_checker = StubGuardrail
    sample = REPO / "data" / "eval" / "sample5"
    clip_id = json.loads((sample / "sample.json").read_text())["clips"][0]["clip_id"]
    with tempfile.TemporaryDirectory() as tmp:
        model, out = Path(tmp) / "tiny-cosmos", Path(tmp) / "pred"
        build_tiny(model)
        predict.main(["--sample", str(sample), "--out", str(out), "--model-id", str(model),
                      "--size", "64", "--num-frames", "21", "--steps", "2", "--clips", clip_id])
        meta = json.loads((out / f"{clip_id}.json").read_text())
        from robust_world.eval.io import read_video
        frames = read_video(out / f"{clip_id}.mp4")
        inputs = read_video(out / f"{clip_id}_input.mp4")
        assert len(frames) == 16 and frames[0].shape == (512, 512, 3), (len(frames), frames[0].shape)
        assert len(inputs) == 5 and meta["input_frame_indices"] == [11, 12, 13, 14, 15]
    print("ok: tiny Cosmos-Predict2 produced 16 target frames from the last 5 context frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
