import os


FEATURE_COLS = [
    'shots_p90', 'xg_p90', 'key_passes_p90', 'pass_accuracy',
    'carries_p90', 'dribble_success_rate', 'interceptions_p90',
    'duel_win_rate', 'recoveries_p90', 'mean_x', 'mean_y', 'activity_spread'
]

LATENT_DIM = 6
HIDDEN_DIM = 16
DEFAULT_TOP_K = 5

EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 1e-3

PROCESSED_FEATURES_PATH = "artifacts/features/processed_features.parquet"
MODEL_PATH = f"artifacts/model/autoencoder_best_{LATENT_DIM}d.pt"
EMBEDDING_TABLE_PATH = f"artifacts/embeddings/player_embedding_table_{LATENT_DIM}d.parquet"
EMBEDDING_NPY_PATH = f"artifacts/embeddings/player_embeddings_{LATENT_DIM}d.npy"

DEFAULT_FEATURE_SOURCE_MODE = os.getenv("TRAIN_FEATURE_SOURCE_MODE", "local")
TRAIN_BQ_PROJECT = os.getenv("TRAIN_BQ_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", "big-data-488609"))
TRAIN_BQ_FEATURE_TABLE = os.getenv("TRAIN_BQ_FEATURE_TABLE", "feature.v_dashboard_players")
