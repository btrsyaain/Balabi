import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List

class NeuralCleanseDefense:
    """Offline trigger inversion + online mitigation for VFL top‑model.

    The class follows the design proposed in *Neural Cleanse: Identifying and
    Mitigating Backdoor Attacks in Neural Networks* (S&P 2019) but is adapted
    to work **purely in the concatenated‑embedding space** produced by a VFL
    server.
    
    Usage
    -----
    >>> nc = NeuralCleanseDefense(args, top_model, device)
    >>> nc.fit(H_train_tensor, y_train_tensor)   # offline after main training
    >>> logits = top_model(H_test)
    >>> logits = nc.filter_and_correct(H_test, logits)  # online mitigation
    """

    # ---------------------------------------------------------------------
    #  Construction
    # ---------------------------------------------------------------------
    def __init__(self, args, top_model: torch.nn.Module, device: torch.device):
        self.top_model = top_model.eval()           # 推論模式
        for p in self.top_model.parameters():       # ★ 凍結權重
            p.requires_grad_(False)
        self.device = device

        # ---- Hyper‑parameters (exposed via CLI) ----
        self.nc_lambda  = getattr(args, 'nc_lambda', 3e-3)
        self.nc_steps   = getattr(args, 'nc_steps', 500)
        self.nc_mad_T   = getattr(args, 'nc_mad_T', 2.5)
        self.nc_batch   = getattr(args, 'nc_batch', 128)
        self.trigger_lr = 0.1                        # internal default

        self.num_classes = args.num_class
        self.suspect: Dict[int, Dict[str, torch.Tensor]] = {}
        # keyed by label → {'mask':…, 'delta':…, 'thresh':…}

    # ------------------------------------------------------------------
    #  (A)  Offline detection  – run once after training
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _score_distribution(self, H: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Compute activation score distribution of clean data for a given trigger.

        score = ‖ (H + mask*delta) − H ‖₁  restricted to *masked* dims
        """
        perturbed = H + mask * delta
        diff = torch.abs((perturbed - H) * (mask > 1e-6))
        return diff.sum(dim=1)                       # (N,)

    def fit(self, H_train: torch.Tensor, y_train: torch.Tensor):
        """Reverse‑engineer candidate triggers for **each** class and locate outliers.

        Parameters
        ----------
        H_train : (N, D) tensor
            Concatenated embeddings of the *clean* validation set collected
            after the final training epoch (already on ``self.device`` is fine).
        y_train : (N,) tensor – ground‑truth labels for H_train.
        """
        H_train = H_train.to(self.device)
        y_train = y_train.to(self.device)
        N, D = H_train.shape

        l1_per_class: List[float] = []
        trig_per_class: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # ---------------- Reverse‑engineer each label ----------------
        for tgt in range(self.num_classes):
            # --- build mini‑dataset for current optimisation cycle
            idx_other = (y_train != tgt).nonzero(as_tuple=True)[0]
            if len(idx_other) > self.nc_batch:
                idx_other = idx_other[:self.nc_batch]
            H_batch = H_train[idx_other]             # (M, D)

            # --- initial mask+delta (trainable) ---
            mask  = torch.zeros(D, device=self.device, requires_grad=True)
            delta = torch.zeros(D, device=self.device, requires_grad=True)
            optim = torch.optim.Adam([mask, delta], lr=self.trigger_lr)

            # --- optimisation loop (safe version) -----------------------
            for _ in range(self.nc_steps):
                # 1. forward
                H_adv  = H_batch + torch.tanh(mask) * delta
                logits = self.top_model(H_adv)

                tgt_vec  = torch.full((logits.size(0),), tgt,
                                    device=self.device, dtype=torch.long)
                ce_loss  = F.cross_entropy(logits, tgt_vec)
                sparsity = self.nc_lambda * torch.abs(torch.tanh(mask)).sum()
                loss     = ce_loss + sparsity

                # 2. 直接取得梯度（不會留下舊計算圖）
                grad_mask, grad_delta = torch.autograd.grad(
                    loss, (mask, delta), retain_graph=False)

                # 3. 手動寫回 grad、更新參數
                mask.grad  = grad_mask
                delta.grad = grad_delta
                optim.step()
                optim.zero_grad()



        # ---------------- Statistical outlier detection ----------------
        l1_tensor = torch.tensor(l1_per_class)
        median = l1_tensor.median()
        mad = torch.median(torch.abs(l1_tensor - median)) + 1e-9
        anomaly_idx = (median - l1_tensor) / mad

        for cls, score in enumerate(anomaly_idx):
            if score > self.nc_mad_T:
                mask, delta = trig_per_class[cls]
                # empirical threshold for runtime detection
                clean_scores = self._score_distribution(H_train, mask, delta)
                trig_thresh = clean_scores.mean() - 3 * clean_scores.std()
                self.suspect[cls] = {
                    'mask':  mask,
                    'delta': delta,
                    'thresh': trig_thresh.item()
                }

        if len(self.suspect) == 0:
            print("[NeuralCleanse] No suspicious classes detected – model seems clean.")
        else:
            print(f"[NeuralCleanse] Detected suspicious classes: {list(self.suspect.keys())}")

    # ------------------------------------------------------------------
    #  (B)  Online mitigation  – call at inference time
    # ------------------------------------------------------------------
    def _is_triggered(self, h: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a boolean mask (batch,) marking which samples are triggered."""
        mask, delta, thresh = info['mask'], info['delta'], info['thresh']
        diff = torch.abs(((h + mask * delta) - h) * (mask > 1e-6)).sum(dim=1)
        return diff < thresh

    @torch.no_grad()
    def filter_and_correct(self, H_in: torch.Tensor, logits_in: torch.Tensor) -> torch.Tensor:
        """Detect triggered samples and patch their logits on‑the‑fly.

        Strategy
        ---------
        1. For every *suspect* label, compute activation score.
        2. If a sample is flagged, **zero‑out** its logit for that target label, so
           the next highest class prevails (equivalent to rejection of the backdoor).
        """
        if not self.suspect:
            return logits_in                                  # model deemed clean

        logits = logits_in.clone()
        H_in = H_in.to(self.device)

        # (1) build a mask of shape (B,) per suspect class
        B = H_in.size(0)
        triggered_any = torch.zeros(B, dtype=torch.bool, device=self.device)
        for cls, info in self.suspect.items():
            tr = self._is_triggered(H_in, info)
            triggered_any |= tr
            # (2) For flagged samples → suppress the malicious logit
            logits[tr, cls] = -1e9                           # a very low value

        return logits
