# vflmonitor.py  ── VFLMonitor defence (training bank + inference majority vote)
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
import random
import math

# ---------- 1.  Full-list Feature-bank builder ----------
class FeatureBankBuilder:
    """
    逐樣本存入 list，最後 concat 成 (N_k, d)。
    pros : 保留類別內多模態資訊，SWD 更準確
    cons : 佔記憶體 (大小 ≈ Σ_k N_k·d·4bytes)
    """
    def __init__(self, num_parties: int, num_classes: int):
        self.num_parties  = num_parties
        self.num_classes  = num_classes
        # bank[p][c] = list[Tensor(d,)]
        self.bank = [ [ [] for _ in range(num_classes) ] for _ in range(num_parties) ]

    @torch.no_grad()
    def update(self, party_embeds, labels):
        """
        party_embeds : List[Tensor]  len = P,  each (B, d)
        labels       : Tensor (B,)
        """
        B = labels.size(0)
        for p in range(self.num_parties):
            emb_p = party_embeds[p]              # (B,d)
            for b in range(B):
                cls = int(labels[b])
                self.bank[p][cls].append(emb_p[b].cpu())   # 存單筆 (d,)

    # def finalize(self):
    #     """
    #     將 list → (N_k, d) tensor；若該類別沒樣本則回空 tensor。
    #     """
    #     full_bank = []
    #     for p in range(self.num_parties):
    #         party_dict = {}
    #         for c in range(self.num_classes):
    #             lst = self.bank[p][c]
    #             if len(lst) == 0:
    #                 party_dict[c] = torch.zeros(0)
    #             else:
    #                 party_dict[c] = torch.stack(lst, dim=0)   # (N_k, d)
    #         full_bank.append(party_dict)
    #     return full_bank

    def finalize(self, max_per_class: int = 2000):
        full_bank = []
        for p in range(self.num_parties):
            party_dict = {}
            for c in range(self.num_classes):
                lst = self.bank[p][c]          # list[Tensor(d,)]
                if len(lst) == 0:
                    party_dict[c] = torch.zeros(0)
                else:
                    # ★ 只隨機留 max_per_class 筆
                    if len(lst) > max_per_class:
                        idx = torch.randperm(len(lst))[:max_per_class]
                        lst = [lst[i] for i in idx]
                    party_dict[c] = torch.stack(lst, dim=0)  # (N', d)
            full_bank.append(party_dict)
        return full_bank

    def save(self, path):
        torch.save(self.finalize(), path)

# ---------- 1.  Memory-efficient Feature-bank builder ----------
# class FeatureBankBuilder:
#     """
#     線上平均、省記憶體版本  
#     bank[p][c] = {'sum': Tensor(d,), 'cnt': int}
#     """
#     def __init__(self, num_parties: int, num_classes: int, device="cpu"):
#         self.num_parties = num_parties
#         self.num_classes = num_classes
#         self.device = device
#         # 建立巢狀 dict
#         self.bank = [ [ {"sum": None, "cnt": 0} for _ in range(num_classes) ]
#                       for _ in range(num_parties) ]

#     @torch.no_grad()
#     def update(self, party_embeds, labels):
#         """
#         party_embeds: List[Tensor] len=P, each (B, d)  (已在 train_new.py .cpu())
#         labels      : Tensor (B,)
#         """
#         B = labels.size(0)
#         for p in range(self.num_parties):
#             emb_party = party_embeds[p]                  # (B, d)
#             for b in range(B):
#                 y = int(labels[b])
#                 vec = emb_party[b]
#                 slot = self.bank[p][y]
#                 if slot["sum"] is None:                  # 第一次出現
#                     slot["sum"] = vec.clone()
#                 else:
#                     slot["sum"] += vec
#                 slot["cnt"] += 1

#     def finalize(self):
#         """
#         將 sum/cnt → mean 向量；若 cnt==0 回傳空 tensor
#         """
#         mean_bank = []
#         for p in range(self.num_parties):
#             party_dict = {}
#             for c in range(self.num_classes):
#                 slot = self.bank[p][c]
#                 if slot["cnt"] == 0:
#                     party_dict[c] = torch.zeros(0, device=self.device)
#                 else:
#                     party_dict[c] = (slot["sum"] / slot["cnt"]).to(self.device)
#             mean_bank.append(party_dict)
#         return mean_bank

#     def save(self, path):
#         torch.save(self.finalize(), path)


# ---------- 2.  SWD util ----------
def sliced_wasserstein_distance(x: torch.Tensor, ref: torch.Tensor,
                                proj_num: int = 128) -> float:
    """
    x : (d,)            ref : (N, d)
    先隨機產生 L 個 d 維單位投影向量，計算 1-d wasserstein，再取平均。
    為求效率，d<=1024 時使用純 torch；proj_num 可透過 args 調小加速。
    """
    if ref.numel() == 0:            # bank 沒資料 → 返回大距離
        return 1e9
    if ref.dim() == 1:              # (d,) → (1, d)   ★ 這行是關鍵修補
        ref = ref.unsqueeze(0)
    d = x.size(0)
    R = torch.randn(proj_num, d, device=x.device)
    R = F.normalize(R, dim=1)                   # (L, d)
    proj_x  = (R @ x)                           # (L,)
    proj_ref = (R @ ref.t()).sort(dim=1).values # (L, N)
    proj_x   = proj_x.unsqueeze(1)              # (L, 1)
    # 1-d Wasserstein = mean |cdf_x − cdf_ref|, 取 N 個點近似
    cdf_ref = torch.linspace(0, 1, ref.size(0), device=x.device)
    cdf_x   = torch.ones_like(cdf_ref) * 0.5    # delta function at proj_x → cdf 0/1
    wasserstein = (proj_ref - proj_x).abs().mean(dim=1)             # (L,)
    return wasserstein.mean().item()

# ---------- 3.  Inference-time defence ----------
class VFLMonitorDefense:
    def __init__(self, bank_path, proj_num=128, majority_ratio=0.5, device="cpu",
                 tie_rand_seed=None):
        self.bank = torch.load(bank_path, map_location=device)
        self.P = len(self.bank)
        self.C = len(self.bank[0])
        self.proj = proj_num
        self.majority = math.floor(self.P * majority_ratio) + 1     # >50%
        if tie_rand_seed is not None:
            random.seed(tie_rand_seed)

    def detect(self, sample_party_embeds):
        """
        sample_party_embeds: List[Tensor] len=P, each (d,)
        回傳 dict：
            decided      : bool  是否有過半共識
            final_label  : int   (decided==True 時有效)
            rand_label   : int   若 undecided，從最高票隨機選
        """
        preds = []
        for p, emb in enumerate(sample_party_embeds):
            dists = [sliced_wasserstein_distance(emb, self.bank[p][k], self.proj)
                     for k in range(self.C)]
            preds.append(int(min(range(self.C), key=lambda k: dists[k])))

        vote_count = Counter(preds).most_common()
        top_label, top_freq = vote_count[0]

        if top_freq >= self.majority:                   # 成功多數決
            return {"decided": True, "final_label": top_label, "rand_label": None}
        else:                                           # tie → 隨機補決策
            max_freq = top_freq
            candidates = [lab for lab, cnt in vote_count if cnt == max_freq]
            rand_label = random.choice(candidates)
            return {"decided": False, "final_label": None, "rand_label": rand_label}
