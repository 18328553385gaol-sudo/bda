from data_loader import load_local_features
from recommender import build_similarity_matrices, recommend_players,recruit_players
from preprocess import build_dimension_scores
from recommender import recommend_players_embedding
import pandas as pd

def main():
    df = load_local_features()
    print("✅ Data loaded")

    feature_cols = [
        'shots_p90', 'xg_p90', 'key_passes_p90', 'pass_accuracy',
        'carries_p90', 'dribble_success_rate', 'interceptions_p90',
        'duel_win_rate', 'recoveries_p90', 'mean_x', 'mean_y', 'activity_spread'
    ]

    sim_matrices = build_similarity_matrices(df, feature_cols)

    test_player_id = str(df.iloc[0]["player_key"])
    print("Test player:", df.iloc[0]["player_name"], "| Position:", df.iloc[0]["position_group"])

    print("\n=== Similar Players (same position only) ===")
    result_filtered = recommend_players(
        df=df,
        sim_matrices=sim_matrices,
        player_id=test_player_id,
        method="cosine",
        same_position_only=True
    )
    print(result_filtered)

    print("\n=== Similar Players (all positions) ===")
    result_all = recommend_players(
        df=df,
        sim_matrices=sim_matrices,
        player_id=test_player_id,
        method="cosine",
        same_position_only=False
    )
    print(result_all)
    
    df_rank = build_dimension_scores(df)

    print("\n=== Recruitment Ranking ===")
    recruit_df = recruit_players(
        df=df_rank,
        target_position=df.iloc[0]["position_group"]
    )

    print(recruit_df[["player_name", "position_group", "recruitment_score"]])
    
    print("\n=== Embedding-based Similar Players ===")
    embedding_df = pd.read_parquet("artifacts/embeddings/player_embedding_table_6d.parquet")
    embed_result = recommend_players_embedding(
        embedding_df,
        player_id=test_player_id,
        top_k=5
    )
    print(embed_result)
    
    print("\n=== Comparison ===")
    print("Baseline vs Embedding overlap:")

    baseline_ids = set(result_filtered["player_key"].astype(str))
    embedding_ids = set(embed_result["player_key"].astype(str))

    print("Overlap:", len(baseline_ids & embedding_ids))
    print("Overlap player IDs:", baseline_ids & embedding_ids)
    
    print("\n=== Baseline vs Embedding Comparison ===")


if __name__ == "__main__":
    main()