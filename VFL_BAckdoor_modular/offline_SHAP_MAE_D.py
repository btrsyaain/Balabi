from dualgate_utils import compute_cos_robust_stats
from train_new import train_SHAP_MAE_offline   # ← 這是您既有的

def train_SHAP_MAE_D_offline(defense_model,
                             concat_embeddings, shapley_mat, labels, args):
    """一步包辦：先訓練 MAE，再補 cosine 統計。"""
    train_SHAP_MAE_offline(defense_model, concat_embeddings, args)   # ① 舊流程
    compute_cos_robust_stats(defense_model, concat_embeddings, labels)    # ② 新統計
