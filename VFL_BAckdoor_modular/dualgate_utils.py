import torch, torch.nn.functional as F
import torch

# ---------- robust helpers ----------
def _median(t, dim=0): return t.median(dim=dim, keepdim=False).values
def _mad(t, dim=0):
    med = _median(t, dim)
    return (t - med.unsqueeze(dim)).abs().median(dim, False).values + 1e-6

# ---------- cosine distance ----------
def cos_distance(orig, rec):       # orig,rec (N,P,D)
    return 1 - F.cosine_similarity(orig, rec, dim=2)   # (N,P)

# # ---------- 離線 robust 統計 ----------
# @torch.no_grad()
# def compute_cos_robust_stats(def_model, concat_emb, labels):
#     """填入 def_model.cos_med / cos_mad  (C,P)"""
#     P, D   = def_model.args.P, def_model.args.bottomModel_output_dim
#     C      = def_model.args.num_class
#     B    = concat_emb.size(0)
    
#     # # rec    = def_model(concat_emb).view(-1, P, D)
#     # rec  = def_model(concat_emb, [[] for _ in range(B)])        # 👈 加空遮罩
#     # cosmat = cos_distance(concat_emb.view(-1,P,D), rec)          # (N,P)
    
#     # ---- 取得 MAE 重建並 reshape ----
#     rec = def_model(concat_emb, [[] for _ in range(B)])           # (N, P*D)
#     rec = rec.view(B, P, D)                                       #  ←★★ 這行補上

#     cosmat = cos_distance(concat_emb.view(B, P, D), rec)          # (N, P)
    
#     for c in range(C):
#         idx = labels == c
#         if idx.sum()==0: continue
#         def_model.cos_med[c] = _median(cosmat[idx], 0)
#         def_model.cos_mad[c] = _mad   (cosmat[idx], 0)
        
# ---------- 離線 robust 統計 ----------
@torch.no_grad()
def compute_cos_robust_stats(def_model, concat_emb, labels):
    """
    依『真實類別』計算 (C,P) 的 cosine  median / MAD，
    並存回 def_model.cos_med / cos_mad
    """
    device = concat_emb.device
    B, P   = labels.size(0), def_model.args.P
    D      = def_model.args.bottomModel_output_dim
    C      = def_model.args.num_class

    # ① --- 取得 *純 AE 重建*（需自行做標準化／反標準化） ----------
    x_norm   = (concat_emb - def_model.mu) / def_model.std          # 標準化
    latent   = def_model.encoder(x_norm)
    rec_norm = def_model.decoder(latent)
    rec      = rec_norm * def_model.std + def_model.mu              # 反標準化
    # rec, concat_emb : (B, P*D)

    # ② --- 每個被動方的 cosine-distance（local reconstruction error）----
    v_orig = concat_emb.view(B, P, D)                               # (B,P,D)
    v_pred = rec       .view(B, P, D)
    cosmat = 1 - F.cosine_similarity(v_orig, v_pred, dim=2)         # (B,P)

    # ③ --- 依類別計 median / MAD -----------------------------------------
    cos_med = torch.zeros(C, P, device=device)
    cos_mad = torch.zeros_like(cos_med)
    for c in range(C):
        idx = (labels == c)
        if idx.any():
            v      = cosmat[idx]                        # (Nc,P)
            med    = v.median(dim=0).values
            mad    = (v - med).abs().median(dim=0).values
            cos_med[c] = med
            cos_mad[c] = mad.clamp(min=1e-3)            # 防止 0 造成門檻鎖死

    # ④ --- 寫回 buffer ------------------------------------------------------
    def_model.cos_med.copy_(cos_med)
    def_model.cos_mad.copy_(cos_mad)

    print(f"[INFO] cos_med  range: {cos_med.min():.4f} ~ {cos_med.max():.4f}")
    print(f"[INFO] cos_mad  range: {cos_mad.min():.4f} ~ {cos_mad.max():.4f}")


# # ---------- dual-gate 遮罩 ----------
# def make_dualgate_mask(
#         shap, cos, cls_idx, act_pid,
#         *,                         # 之後皆 keyword-only
#         shap_up=None, shap_down=None,           # 新加：若給定就用它們
#         shap_med=None, shap_mad=None,           # 退而求其次
#         cos_med=None,  cos_mad=None,
#         k_shap=1.0, k_cos=1.0,
#         return_counts=True):                 # 可微調門檻
#     """
#     回傳 Boolean mask (B,P)：
#         · 若 active-party Shapley 異常偏低 → 需同時滿足 S¹ & S²
#         · 否則只檢查 S² (cos)
#         · active_pid 永遠為 False
#     """
#     B, P = shap.shape
#     device = shap.device
#     cls_idx = cls_idx.to(torch.long)

#     # --- S¹：Shapley gate --------------------------------------------------
#     if shap_up is not None and shap_down is not None:
#         # shap_up / shap_down 已是 (B,P) or (num_cls,P) 的閾值
#         s1_bool = (shap > shap_up) | (shap < shap_down)
#         active_low = shap[torch.arange(B), act_pid] < shap_down[torch.arange(B), act_pid]
#     else:
#         # 後備方案：median ± k·MAD
#         assert shap_med is not None and shap_mad is not None
#         med_b = shap_med[cls_idx]            # (B,P)
#         mad_b = shap_mad[cls_idx]
#         s1_bool   = (shap - med_b).abs() > k_shap * mad_b
#         active_low = shap[torch.arange(B), act_pid] < (med_b[:, act_pid] - k_shap * mad_b[:, act_pid])

#     # --- S²：Cosine gate ----------------------------------------------------
#     if cos_med is not None and cos_mad is not None:
#         cos_th_b = cos_med[cls_idx] + k_cos * cos_mad[cls_idx]
#         s2_bool  = cos > cos_th_b
#     else:                                         # 若不想用 cos gate
#         s2_bool  = torch.ones_like(shap, dtype=torch.bool, device=device)

#     # --- 組合條件 -----------------------------------------------------------
#     mask = torch.where(active_low.unsqueeze(1),   # (B,1) broadcast
#                        s1_bool & s2_bool,         # active 低 → AND
#                        s2_bool)                   # 否則只看 S²
#     mask[:, act_pid] = False                      # active party 永遠保留

#     if return_counts:
#         # 統計「有幾個 (b,p) 為 True」
#         shap_cnt = int(s1_bool.sum().item())   # S¹ 觸發次數
#         cos_cnt  = int(s2_bool.sum().item())   # S² 觸發次數
#         return mask, shap_cnt, cos_cnt
#     else:
#         return mask, None, None

# # ---------- dual-gate 遮罩（re-worked） ----------
# def make_dualgate_mask(
#         shap, cos, cls_idx, act_pid,
#         *,                         # 之後皆 keyword-only
#         shap_up=None, shap_down=None,           # 可選：(C,P) or (B,P)
#         shap_med=None, shap_mad=None,           # 後備：median / MAD
#         cos_med=None,  cos_mad=None,
#         k_shap=1.0, k_cos=0.01,
#         return_counts=False):

#     """
#     回傳 Boolean mask (B,P)：

#       ● 若 active-party 的 Shapley 〈 shap_down →  
#         ⓵ 先找 (shap > shap_up) AND (cos > thr)；  
#         ⓶ 若該列沒有任何 AND 命中 → 改用 OR。  
#       ● 其他情況僅看 (cos > thr)。  
#       ● active_pid 永不遮。  
#     """

#     B, P = shap.shape
#     dev  = shap.device
#     cls_idx = cls_idx.long()

#     # ---------- S¹：Shapley gate（只取「高」） ----------
#     if shap_up is not None:                                      # 已提供門檻
#         s1_hi = shap > shap_up                                   # (B,P)
#     else:                                                        # 後備：median + k·MAD
#         assert shap_med is not None and shap_mad is not None
#         med_b = shap_med[cls_idx]                                # (B,P)
#         mad_b = shap_mad[cls_idx]
#         s1_hi = (shap - med_b) > k_shap * mad_b                  # 只看「高」

#     # ---------- 判斷 active_party 是否「低於」 shap_down ----------
#     if shap_down is not None:
#         active_low = shap[:, act_pid] < shap_down[:, act_pid]    # (B,)
#     else:
#         med_a = shap_med[cls_idx, act_pid]
#         mad_a = shap_mad[cls_idx, act_pid]
#         active_low = shap[:, act_pid] < (med_a - k_shap * mad_a)

#     # ---------- S²：Cosine gate ----------
#     if cos_med is not None:
#         cos_thr = cos_med[cls_idx] + k_cos * cos_mad[cls_idx]
#         s2_hi   = cos > cos_thr
#     else:                                                        # 不想用 cos gate
#         s2_hi   = torch.ones_like(shap, dtype=torch.bool, device=dev)

#     # ---------- 先計算 AND / OR ------------
#     both_hi  = s1_hi & s2_hi
#     union_hi = s1_hi | s2_hi

#     # ---------- 組合條件 ----------
#     mask = torch.where(active_low.unsqueeze(1),   # (B,1) broadcast
#                        both_hi,                   # 先採 AND
#                        s2_hi)                     # 否則僅看 cos

#     # 若 active_low & AND 全 False → fallback 取 OR
#     need_fallback = active_low & (~both_hi.any(dim=1))
#     if need_fallback.any():
#         mask[need_fallback] = union_hi[need_fallback]

#     # active party 永遠不遮
#     mask[:, act_pid] = False

#     # ---------- 統計（可選） ----------
#     if return_counts:
#         return mask, int(s1_hi.sum()), int(s2_hi.sum())
#     else:
#         return mask

# ---------- dual-gate 遮罩 ----------
def make_dualgate_mask(
        shap, cos, cls_idx, act_pid,
        *,                         # 之後皆 keyword-only
        shap_up=None, shap_down=None,
        shap_med=None, shap_mad=None,
        cos_med=None,  cos_mad=None,
        k_shap=1.0, k_cos=1.0,
        return_counts=True):

    """
    回傳 Boolean mask (B,P)：
        1. 若某筆樣本存在「同時」滿足 S¹& S² 的參與者 → 只遮蔽這些 AND 觸發者
        2. 否則
           • 若 active-party Shapley 異常低 → 遮 (S¹ ∨ S²)
           • 其他樣本                → 只看 S²
        3. active_pid 永遠為 False
    """
    B, P = shap.shape
    device  = shap.device
    arangeB = torch.arange(B, device=device)
    cls_idx = cls_idx.to(torch.long)

    # --- S¹：Shapley gate --------------------------------------------------
    if shap_up is not None and shap_down is not None:
        s1_bool = (shap > shap_up) | (shap < shap_down)                  # (B,P)
        active_low = shap[arangeB, act_pid] < shap_down[arangeB, act_pid]
    else:
        med_b = shap_med[cls_idx]                                        # (B,P)
        mad_b = shap_mad[cls_idx]
        s1_bool   = (shap - med_b).abs() > k_shap * mad_b
        active_low = shap[arangeB, act_pid] < (med_b[:, act_pid] - k_shap * mad_b[:, act_pid])

    # --- S²：Cosine gate ---------------------------------------------------
    if cos_med is not None and cos_mad is not None:
        cos_th_b = cos_med[cls_idx] + k_cos * cos_mad[cls_idx]           # (B,P)
        s2_bool  = cos > cos_th_b
    else:                                                                # 不使用 cos gate
        s2_bool  = torch.zeros_like(shap, dtype=torch.bool, device=device)
    
    # --------- 新增保險條件 ---------------------------------------------
    # 任何 cos_distance > 1.0 的極端情形，一律納入遮蔽
    s2_bool |= (cos > 1.0)              # ← (2) 就在這一行加進去

    # ---------- 新增：AND-first 規則 --------------------------------------
    and_bool = s1_bool & s2_bool                                         # (B,P)
    has_and  = and_bool.any(dim=1)                                       # (B,) 逐樣本

    # 先準備 fallback mask：依舊邏輯
    fallback_mask = torch.where(active_low.unsqueeze(1),                 # (B,1)
                                s1_bool | s2_bool,                      # active 低 → OR
                                s2_bool)                                # 否則只看 S²

    # 若本樣本 has_and=True，改用 and_bool；否則用 fallback_mask
    mask = torch.where(has_and.unsqueeze(1), and_bool, fallback_mask)    # (B,P)

    # active party 永遠保留
    mask[:, act_pid] = False

    # --- 統計 --------------------------------------------------------------
    if return_counts:
        shap_cnt = int(s1_bool.sum().item())
        cos_cnt  = int(s2_bool.sum().item())
        and_cnt  = int(and_bool.sum().item())
        return mask, shap_cnt, cos_cnt, and_cnt
    else:
        return mask, None, None, None