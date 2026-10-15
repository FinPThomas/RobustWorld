"""Sharper ball positions by nearest neighbour in V-JEPA's own feature space (Fin, 2026-10-11: slope map).

The frozen eval_decoder reads the ball per 32 px token cell, so its positions jitter by about 14 px from step to
step (tracker: 2.4 px), far more than the slope moves the ball (about 1 px per step²). This adds the sub-cell part
without training anything: the decoder's peak cell gives the coarse position, and the 3x3 tokens around it are
matched (cosine, layer-normalised tokens) against the same patches from real context frames of the fold's
training clips, where the tracker says where in the cell the ball was. The position is the cell centre plus the
median offset of the nearest K patches.

What it may see is what eval_decoder may see (CLAUDE.md rules 3 and 5): real, context-half features only, each
labelled with the ball position in the same frame. It is built from (context features, context centres) and
nothing else, so outcomes, target labels and predictions can't reach it.
"""

from __future__ import annotations

import numpy as np
import torch

from eval_decoder import token_space

CLIP_SIZE = 512
K = 8
MIN_PEAK = 0.2                   # steps whose decoder map peaks lower get no position (no visible ball)


def patch(tokens: torch.Tensor, cy: int, cx: int) -> torch.Tensor:
    """3x3 layer-normalised tokens ([g, g, D], one step) around (cy, cx), zeros off the grid -> flat unit vector."""
    g, d = tokens.shape[0], tokens.shape[-1]
    out = torch.zeros(3, 3, d)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y, x = cy + dy, cx + dx
            if 0 <= y < g and 0 <= x < g:
                out[dy + 1, dx + 1] = tokens[y, x]
    v = out.reshape(-1)
    return v / (v.norm() + 1e-6)


class Bank:
    """Patches around the ball's cell in real context frames, with the ball's offset from the cell centre."""

    def __init__(self, context: list[tuple[torch.Tensor, list]]):
        keys, offs = [], []
        for feats, centres in context:                 # feats [steps, g, g, D]; centres per context step
            g = feats.shape[1]
            cell = CLIP_SIZE / g
            for t, c in enumerate(centres[:feats.shape[0]]):
                if c is None:
                    continue
                cx, cy = min(int(c[0] // cell), g - 1), min(int(c[1] // cell), g - 1)
                keys.append(patch(token_space(feats[t]), cy, cx).half())
                offs.append((c[0] - (cx + 0.5) * cell, c[1] - (cy + 0.5) * cell))
        self.keys = torch.stack(keys) if keys else torch.zeros(0, 1)
        self.offs = np.array(offs, float).reshape(-1, 2)

    def locate(self, feats: torch.Tensor, maps: np.ndarray) -> list:
        """Per step [x, y, peak] in clip px (None where the map peaks below MIN_PEAK)."""
        out = []
        g = maps.shape[-1]
        cell = CLIP_SIZE / g
        for t in range(len(maps)):
            peak = float(maps[t].max())
            if peak < MIN_PEAK or len(self.offs) < K:
                out.append(None)
                continue
            cy, cx = np.unravel_index(int(maps[t].argmax()), maps[t].shape)
            sim = self.keys.float() @ patch(token_space(feats[t]), int(cy), int(cx))
            near = torch.topk(sim, K).indices.numpy()
            dx, dy = np.median(self.offs[near], axis=0)
            out.append([round(float((cx + 0.5) * cell + dx), 1), round(float((cy + 0.5) * cell + dy), 1),
                        round(peak, 3)])
        return out
