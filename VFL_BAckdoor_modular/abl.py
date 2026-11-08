# abl.py ── Anti‑Backdoor Learning 兩階段 loss 管理器
import torch

class ABLLossManager:
    """
    管理 ABL 的 Local Gradient Ascent (LGA) 與 Global Gradient Ascent (GGA)

    Parameters
    ----------
    gamma : float
        LGA 階段的 loss 閾值 γ
    lowest_percent : float
        用於挑選「最低 loss 百分比」的 p
    switch_epoch : int
        第 switch_epoch 個 epoch 進入 GGA
    """
    def __init__(self,
                 gamma: float = 0.5,
                 lowest_percent: float = 0.01,
                 switch_epoch: int = 20,
                 collect_epoch: int = 20,
                 lambda_bd: int = 0.05) -> None:
        self.gamma = gamma
        self.lowest_percent = lowest_percent
        self.switch_epoch = switch_epoch
        self.collect_epoch = collect_epoch # 收集多少輪
        self._records: list[tuple[int, float]] = []   # (sample_id, loss)
        self.suspect_ids: set[int] = set()
        self.lambda_bd = lambda_bd              # ★ 新增超參 (可寫成 self.lambda_bd)
        print(f"ABL 會在第{self.switch_epoch-self.collect_epoch}輪時開始LGA，在第{self.switch_epoch}輪時開始GGA")

    # -------- 早期 (LGA)：收集每筆 loss --------
    def collect(self,
                sample_ids: torch.Tensor,
                losses: torch.Tensor,
                epoch: int) -> None:
        if (epoch < self.switch_epoch) and (epoch >= self.switch_epoch-self.collect_epoch):
            for sid, l in zip(sample_ids, losses):
                self._records.append((int(sid), float(l)))

    # -------- LGA → GGA 交界：決定可疑樣本 --------
    def finalize_LGA(self, epoch: int) -> None:
        if epoch == self.switch_epoch - 1:
            self._records.sort(key=lambda x: x[1])  # 由小→大
            k = max(1, int(len(self._records) * self.lowest_percent))
            self.suspect_ids = {sid for sid, _ in self._records[:k]}
            self._records.clear()                   # 釋放記憶體

    # -------- 回傳滿足 ABL 規則的 total_loss --------
    def total_loss(self,
                   loss_all: torch.Tensor,
                   sample_ids: torch.Tensor,
                   epoch: int) -> torch.Tensor:
        """
        loss_all  : shape (B,) – 每筆樣本的 Cross‑Entropy
        sample_ids: shape (B,) – 與 batch 對應的整數 id
        """
        # 階段 1：Local Gradient Ascent (LGA)
        if epoch < self.switch_epoch:
            if epoch < self.switch_epoch - self.collect_epoch:
                return loss_all.mean()
            else:
                low_mask  = loss_all < self.gamma
                high_mask = ~low_mask

                loss_lga  = -(loss_all[low_mask] - self.gamma).mean() if low_mask.any() else 0
                loss_rest =  (loss_all[high_mask]).mean()             if high_mask.any() else 0
                return loss_lga + loss_rest

        # 階段 2：Global Gradient Ascent (GGA)
        backdoor_mask = torch.tensor(
            [int(sid) in self.suspect_ids for sid in sample_ids],
            device=loss_all.device, dtype=torch.bool)
        clean_mask = ~backdoor_mask

        loss_clean = loss_all[clean_mask].mean() if clean_mask.any() else 0
        loss_bd    = loss_all[backdoor_mask].mean() if backdoor_mask.any() else 0
        total = loss_clean - self.lambda_bd * loss_bd
        return total
    
'''hyperparameter
==================================
VILLIAN
    percent:
    CINIC 0.003
    CIFAR10 0.01

    gamma:
            CIFAR   CINIC    BM
    2-party  0.5     0.3    0.5
    4-party  0.5     0.3    0.3
    8-party  0.5     0.1    0.1

    --abl_turn_epochs
    25
==================================
TIFS 不適用，會進行2次下毒。
    
'''