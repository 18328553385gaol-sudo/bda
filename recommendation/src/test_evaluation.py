import pandas as pd

from data_loader import load_local_features
from recommender import (
    build_similarity_matrices,
    recommend_players,
    recommend_players_embedding
)
from evaluation import compare_models
from config import FEATURE_COLS, EMBEDDING_TABLE_PATH


# load data
df = load_local_features()
embedding_df = pd.read_parquet(EMBEDDING_TABLE_PATH)

# build baseline similarity
sim_matrices = build_similarity_matrices(df, FEATURE_COLS)

# test player
player_id = "57480"
target_position = "DF"

baseline_result = recommend_players(
    df=df,
    sim_matrices=sim_matrices,
    player_id=player_id,
    top_k=5,
    method="cosine",
    same_position_only=True
)

embedding_result = recommend_players_embedding(
    embedding_df=embedding_df,
    player_id=player_id,
    top_k=5,
    same_position_only=True
)

summary = compare_models(
    baseline_result,
    embedding_result,
    target_position
)

print(summary)