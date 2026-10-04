"""The frozen evaluation decoder: the ONLY trained readout allowed on V-JEPA 2 features.

Evaluation must never be able to learn the blockade. Learning about the scene (including
the blockade) belongs to post-training the world model; the decoder is a measuring
instrument and has to be identical before and after that post-training. So:

  * It is fitted only on REAL encoder features of the CONTEXT half, encoded on its own
    (the encoder never sees the future), labelled with where the ball is in THAT SAME
    frame. The context half ends before the ball reaches the plank, so the decoder never
    sees a ball under or beyond the plank, a blocked ball, or any outcome.
  * It never sees model predictions (imagined features), target-half labels or outcomes.
    `examples_from` is the only way to build its training data and it reads nothing else;
    there is deliberately no parameter through which those could be passed.
  * It is linear (1024 -> 1 per token cell), so whatever it reads out was already in the
    features.

tests/test_eval_isolation.py checks this: scrambling outcomes, target-half labels,
imagined features or full-clip features must leave the fitted decoder bit-for-bit
unchanged.

It reads every token after a fixed, parameter-free layer norm (`token_space`). V-JEPA 2's
predictor was pretrained to output layer-normalised encoder features, while the encoder itself
returns un-normalised ones; without this the decoder would read real and imagined tokens on
different scales (about 5x apart) and see no ball in any prediction. The norm has no fitted
parameters and is the same before and after post-training. `fit_raw_features` is the same decoder
without that norm (the original instrument), for the alternative of mapping predictions into encoder space instead
(gamma * prediction + beta, with the encoder's own final layer norm).

Even so, it finds the ball beyond the plank in real target frames (AUROC ~0.997), so it is
not blind to the region the outcome is read from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EvalExample:
    """Training data for the decoder from one clip: context-half features and same-frame labels."""
    context_features: torch.Tensor     # [ctx_steps, g, g, D], encoder run on the context half alone
    context_labels: np.ndarray         # [ctx_steps, g, g] bool, ball cells in those same frames


def examples_from(encoding: dict, labels: np.ndarray) -> EvalExample:
    """Take only the context-only encoding and the context steps' labels; everything else is dropped."""
    ctx = encoding["context"]
    return EvalExample(context_features=ctx, context_labels=np.asarray(labels[:ctx.shape[0]], bool))


def token_space(tokens: torch.Tensor) -> torch.Tensor:
    """Layer-normalise each token (no learned scale): the space V-JEPA 2's predictor outputs."""
    return torch.nn.functional.layer_norm(tokens.float(), tokens.shape[-1:])


class EvalDecoder:
    """Calibrated linear map from a token's features to P(ball in this cell)."""

    def __init__(self, linear: torch.nn.Linear, mu: torch.Tensor, sd: torch.Tensor, normalise: bool = True):
        self.linear, self.mu, self.sd, self.normalise = linear, mu, sd, normalise

    def _space(self, tokens: torch.Tensor) -> torch.Tensor:
        return token_space(tokens) if self.normalise else tokens.float()

    @torch.no_grad()
    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:          # [..., D] -> [...]
        return torch.sigmoid(self.linear((self._space(tokens) - self.mu) / self.sd).squeeze(-1))

    def state(self) -> dict:
        return {"weight": self.linear.weight.detach().clone(), "bias": self.linear.bias.detach().clone(),
                "mu": self.mu.clone(), "sd": self.sd.clone()}


def fit(examples: list[EvalExample], seed: int = 0, epochs: int = 20) -> EvalDecoder:
    """Fit on context-half examples only. Plain BCE keeps outputs calibrated (background ~0)."""
    return _fit(examples, seed, epochs, normalise=True)


def fit_raw_features(examples: list[EvalExample], seed: int = 0, epochs: int = 20) -> EvalDecoder:
    """The same decoder on raw encoder features (no layer norm): reads encoder-space tokens only."""
    return _fit(examples, seed, epochs, normalise=False)


def _fit(examples: list[EvalExample], seed: int, epochs: int, normalise: bool) -> EvalDecoder:
    if not examples or not all(isinstance(e, EvalExample) for e in examples):
        raise TypeError("fit() takes EvalExample objects built with examples_from()")
    d = examples[0].context_features.shape[-1]
    x = torch.cat([e.context_features.reshape(-1, d) for e in examples])
    x = token_space(x) if normalise else x.float()
    y = torch.cat([torch.from_numpy(e.context_labels.reshape(-1)) for e in examples]).float()

    torch.manual_seed(seed)
    mu, sd = x.mean(0), x.std(0) + 1e-4
    lin = torch.nn.Linear(d, 1)
    rate = y.mean().clamp(1e-4, 1 - 1e-4)
    with torch.no_grad():                    # start at the base rate (~1.5% of cells), not 50%
        lin.weight.zero_()
        lin.bias.fill_(float(torch.log(rate / (1 - rate))))
    opt = torch.optim.AdamW(lin.parameters(), lr=3e-3, weight_decay=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for idx in torch.randperm(len(x)).split(4096):
            opt.zero_grad()
            loss_fn(lin((x[idx] - mu) / sd).squeeze(1), y[idx]).backward()
            opt.step()
    return EvalDecoder(lin, mu, sd, normalise)
