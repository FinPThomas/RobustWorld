# LTX-Video 2B distilled

Fast video continuation with LTX-Video 0.9.8 2B distilled, through Diffusers. It's
self-contained and doesn't use any other model folder.

**How it predicts.**
- LTX's VAE compresses time 8×, so a conditioning video must be 8k+1 frames long. The
  pipeline trims longer ones *from the start*.
- So it receives the last 9 context frames (7–15) and generates 25 frames, covering
  clip frames 7–31.
- Generated frames 9–24 are compared with ground-truth frames 16–31.
- The distilled model uses 8 fixed timesteps and no classifier-free guidance, which is
  why it's fast.

**Weights.**
- The transformer and VAE come from the single file
  `Lightricks/LTX-Video/ltxv-2b-0.9.8-distilled.safetensors` (6.3 GB, bf16). Its tensors
  match Diffusers' 0.9.5 architecture exactly.
- The T5 text encoder, tokenizer and distilled scheduler come from
  `Lightricks/LTX-Video-0.9.7-distilled`. Only those subfolders are downloaded.
- No Hugging Face token is needed.

**Run on Colab.** Open [`notebooks/colab_ltx_video.ipynb`](../../notebooks/colab_ltx_video.ipynb).
A free T4 is enough.

**Run anywhere with a CUDA GPU:**
```bash
pip install -e . -r models/ltx_video/requirements.txt
python models/ltx_video/predict.py --sample data/eval/sample5 --out outputs/predictions/sample5/ltx_video
robustworld-eval grid --models ltx_video
```

**Plumbing test without a GPU.** `python models/ltx_video/test_tiny.py`.
