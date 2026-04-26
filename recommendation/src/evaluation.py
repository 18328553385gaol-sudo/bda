import pandas as pd


def overlap_at_k(df1, df2):
    """
    Top-K 推荐结果重合数量
    """
    ids1 = set(df1["player_key"].astype(str))
    ids2 = set(df2["player_key"].astype(str))

    overlap_ids = ids1 & ids2

    return {
        "overlap_count": len(overlap_ids),
        "overlap_ids": list(overlap_ids)
    }


def position_purity_at_k(result_df, target_position):
    """
    推荐结果中，同位置球员比例
    """
    if len(result_df) == 0:
        return 0.0

    same_position_count = (
        result_df["position_group"] == target_position
    ).sum()

    return same_position_count / len(result_df)


def avg_similarity_at_k(result_df):
    """
    Top-K 平均相似度
    """
    if len(result_df) == 0:
        return 0.0

    return result_df["similarity_score"].mean()


def compare_models(
    baseline_df,
    embedding_df,
    target_position
):
    """
    汇总 baseline vs embedding 的评估结果
    """

    overlap_result = overlap_at_k(baseline_df, embedding_df)

    baseline_purity = position_purity_at_k(
        baseline_df, target_position
    )

    embedding_purity = position_purity_at_k(
        embedding_df, target_position
    )

    baseline_avg_sim = avg_similarity_at_k(baseline_df)
    embedding_avg_sim = avg_similarity_at_k(embedding_df)

    summary = pd.DataFrame({
        "metric": [
            "overlap_count",
            "baseline_position_purity",
            "embedding_position_purity",
            "baseline_avg_similarity",
            "embedding_avg_similarity"
        ],
        "value": [
            overlap_result["overlap_count"],
            round(baseline_purity, 3),
            round(embedding_purity, 3),
            round(baseline_avg_sim, 3),
            round(embedding_avg_sim, 3)
        ]
    })

    return summary