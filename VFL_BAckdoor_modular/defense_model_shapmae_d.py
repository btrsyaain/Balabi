# defense_shapmae_d.py ── Shapley‑aware MAE (dual‑gate) variant
# This module implements SHAPMAE_D, an experimental defense model that
# follows your new two‑gate decision logic:
#   – Gate‑1 (S1): Shapley absolute deviation beyond MAD
#   – Gate‑2 (S2): Cosine‑distance deviation beyond MAD on MAE‑reconstructed
#                  embeddings
#  • If active party's Shapley is NOT abnormally low (>= median‑MAD),
#    Gate‑1 is skipped and only Gate‑2 is used.
#  • Active party is never mitigated.
#
# The file provides:
#   1. SHAPMAE_D(nn.Module): encoder‑decoder + online purification
#   2. fit_offline(): one‑shot routine to compute per‑class median/MAD of
#      Shapley and cosine distance & to train the MAE weights.
#   3. build_defense(): helper that hides argparse binding details from
#      train_new.py (so train_new remains clean).

from __future__ import annotations
import torch, math, itertools
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

# =====================
#  Utility: robust stat
# =====================

def _median(t: torch.Tensor, dim: int) -> torch.Tensor:
    return t.median(dim=dim, keepdim=False).values

def _mad(t: torch.Tensor, dim: int) -> torch.Tensor:
    med = _median(t, dim)
    return (t - med.unsqueeze(dim)).abs().median(dim=dim, keepdim=False).values + 1e-6  # eps to avoid 0