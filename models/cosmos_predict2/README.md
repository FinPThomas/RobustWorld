# NVIDIA Cosmos-Predict2 Video2World

World-model rollout with
[Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)
through Diffusers. It's self-contained: it doesn't use the Wan code, and this folder can
be deleted without affecting anything else.

**How it predicts.**
- Cosmos-Predict2 was trained to condition on 1 or 5 frames, so it receives the last
  5 context frames (11–15).
- It generates `--num-frames` frames in total, including those 5. Its native length is 93.
- Generated frames 5–20 are compared with ground-truth frames 16–31.
- The diffusers weights are the 720p version. Square clips are therefore run at
  960×960, the 1:1 shape at 720p, and scaled back down to 512.

**Licence and guardrail.**
- Both the model and `nvidia/Cosmos-Guardrail1` are gated on Hugging Face. Accept both
  licences, then provide `HF_TOKEN`.
- The NVIDIA Open Model License forbids disabling the guardrail.
- When the pipeline is given precomputed prompt embeddings it skips its own text check,
  so `predict.py` runs that check explicitly. Every generated video still goes through
  the pipeline's video check.

**Run on Colab.** Open
[`notebooks/colab_cosmos_predict2.ipynb`](../../notebooks/colab_cosmos_predict2.ipynb).
- It needs an A100: NVIDIA lists about 32 GB of GPU memory.
- The default (93 frames at 960×960) is slow. `--num-frames 29` is much faster, but it's
  outside the length the model was trained on.

**Run anywhere with a CUDA GPU:**
```bash
pip install -e . -r models/cosmos_predict2/requirements.txt
python models/cosmos_predict2/predict.py --sample data/eval/sample5 --out outputs/predictions/sample5/cosmos_predict2
robustworld-eval grid --models cosmos_predict2
```

**Plumbing test without a GPU.** `python models/cosmos_predict2/test_tiny.py` builds a
tiny randomly initialised model and replaces the guardrail with a pass-through stub. That
is only acceptable because no NVIDIA weights are involved.
