# Wan 2.1 VACE

Video extension with [Wan 2.1 VACE](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers)
through Diffusers. It's self-contained: it doesn't use the Cosmos code, and this folder
can be deleted without affecting anything else.

**How it predicts.** All 16 context frames are passed in unchanged. 17 grey frames
follow, with a white mask marking them for generation. That makes 33 frames in total,
because Wan needs a frame count of the form 4n+1. Generated frames 16–31 are compared
with the ground-truth target.

**Run on Colab.** Open [`notebooks/colab_wan21.ipynb`](../../notebooks/colab_wan21.ipynb).
- A T4 works and runs fp16. If the output comes back black or noisy, switch to an L4
  or A100, which run bf16.
- The 14B model (`--model-id Wan-AI/Wan2.1-VACE-14B-diffusers`) needs an A100 80 GB.

**Run anywhere with a CUDA GPU:**
```bash
pip install -e . -r models/wan21/requirements.txt
python models/wan21/predict.py --sample data/eval/sample5 --out outputs/predictions/sample5/wan21
robustworld-eval grid --models wan21
```

**Memory.** The prompt is the same for every clip. The UMT5-XXL text encoder (~11 GB)
encodes it once and is freed before the video model loads.

**Plumbing test without a GPU.** `python models/wan21/test_tiny.py` builds a tiny
randomly initialised VACE and checks frame counts and output files.
