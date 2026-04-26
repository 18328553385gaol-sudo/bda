import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def build_similarity_matrix(features, method='cosine'):
    if method == 'cosine':
        return cosine_similarity(features)
    elif method == 'euclidean':
        dist = euclidean_distances(features)
        return 1 / (1 + dist)
    else:
        raise ValueError("method 必须是 'cosine' 或 'euclidean'")


def build_similarity_matrices(df, feature_cols):
    features = df[feature_cols].values

    pca = PCA(n_components=0.9)
    pca_features = pca.fit_transform(features)

    return {
        "cosine": build_similarity_matrix(features, "cosine"),
        "euclidean": build_similarity_matrix(features, "euclidean"),
        "pca": build_similarity_matrix(pca_features, "cosine")
    }

def recruit_players(
    df,
    target_position,
    top_k=5,
    attacking_weight=0.3,
    progression_weight=0.3,
    defensive_weight=0.3,
    spatial_weight=0.1
):
    candidates = df[df["position_group"] == target_position].copy()

    candidates["recruitment_score"] = (
        attacking_weight * candidates["attacking_score"] +
        progression_weight * candidates["progression_score"] +
        defensive_weight * candidates["defensive_score"] +
        spatial_weight * candidates["spatial_score"]
    )

    return (
        candidates.sort_values("recruitment_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

def recommend_players(df, sim_matrices, player_id, top_k=5, method="cosine", same_position_only=True):
    df = df.reset_index(drop=True)

    filtered_df = df[df["player_key"].astype(str) == str(player_id)]
    if filtered_df.empty:
        raise ValueError(f"找不到球员 ID: {player_id}")

    target_row = filtered_df.iloc[0]
    target_idx = filtered_df.index[0]
    target_position = target_row["position_group"]

    sim_matrix = sim_matrices[method]
    scores = list(enumerate(sim_matrix[target_idx]))

    results = []
    for idx, score in scores:
        candidate = df.iloc[idx]

        if idx == target_idx:
            continue

        if same_position_only and candidate["position_group"] != target_position:
            continue

        results.append({
            "player_key": candidate["player_key"],
            "player_name": candidate["player_name"],
            "position_group": candidate["position_group"],
            "similarity_score": float(score)
        })

    return (
        pd.DataFrame(results)
        .sort_values("similarity_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    
def recommend_players_embedding(
    embedding_df,
    player_id,
    top_k=5,
    same_position_only=True
):
    df = embedding_df.reset_index(drop=True)

    # 1️⃣ 找目标球员
    filtered = df[df["player_key"].astype(str) == str(player_id)]
    if filtered.empty:
        raise ValueError(f"找不到球员 ID: {player_id}")

    target_row = filtered.iloc[0]
    target_idx = filtered.index[0]
    target_position = target_row["position_group"]

    # 2️⃣ 取 embedding 向量
    embedding_cols = [col for col in df.columns if col.startswith("embedding_")]
    features = df[embedding_cols].values

    sim_matrix = cosine_similarity(features)
    scores = list(enumerate(sim_matrix[target_idx]))

    # 3️⃣ 遍历 candidate（这里才做 filter）
    results = []
    for idx, score in scores:
        candidate = df.iloc[idx]

        # 排除自己
        if idx == target_idx:
            continue

        # ⭐ 关键：位置过滤
        if same_position_only and candidate["position_group"] != target_position:
            continue

        results.append({
            "player_key": candidate["player_key"],
            "player_name": candidate["player_name"],
            "position_group": candidate["position_group"],
            "similarity_score": float(score)
        })

    # 4️⃣ 排序
    return (
        pd.DataFrame(results)
        .sort_values("similarity_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    
def get_player_options(df):
    """
    返回给前端下拉框使用的球员列表
    """
    options = df[['player_key', 'player_name', 'position_group']].copy()
    options['label'] = options['player_name'] + " | " + options['position_group'] + " | " + options['player_key'].astype(str)
    return options
    