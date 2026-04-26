import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from config import (
    DEFAULT_FEATURE_SOURCE_MODE,
    DEFAULT_TOP_K,
    EMBEDDING_TABLE_PATH,
    FEATURE_COLS,
    TRAIN_BQ_FEATURE_TABLE,
    TRAIN_BQ_PROJECT,
)
from data_loader import load_features
from evaluation import compare_models
from recommender import (
    build_similarity_matrices,
    get_player_options,
    recommend_players,
    recommend_players_embedding,
)


st.set_page_config(
    page_title="Smart Scouting Recommendation System",
    page_icon=":soccer:",
    layout="wide"
)

st.title("Similar Player Recommendation")


def has_streamlit_secrets_file() -> bool:
    secret_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
    ]
    return any(path.exists() for path in secret_paths)


def get_setting(name: str, default: str | None = None) -> str | None:
    if has_streamlit_secrets_file() and name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


def get_feature_source_config() -> dict:
    return {
        "mode": (get_setting("SIMILAR_PLAYER_FEATURE_SOURCE_MODE", DEFAULT_FEATURE_SOURCE_MODE) or "local").lower(),
        "local_path": get_setting("SIMILAR_PLAYER_LOCAL_FEATURE_PATH", get_setting("PROCESSED_FEATURES_PATH", "artifacts/features/processed_features.parquet")),
        "bq_project": get_setting("SIMILAR_PLAYER_BQ_PROJECT", TRAIN_BQ_PROJECT),
        "bq_table": get_setting("SIMILAR_PLAYER_BQ_TABLE", TRAIN_BQ_FEATURE_TABLE),
        "embedding_path": get_setting("SIMILAR_PLAYER_EMBEDDING_PATH", EMBEDDING_TABLE_PATH),
    }


def show_data_source_help(config: dict):
    with st.sidebar:
        st.subheader("Feature Source")
        st.caption(f"Current mode: `{config['mode']}`")
        if config["mode"] == "bigquery":
            st.code(f"{config['bq_project']}.{config['bq_table']}")
        else:
            st.code(config["local_path"])

        st.subheader("Embedding Source")
        st.code(config["embedding_path"])

        st.markdown(
            """
            Set these env vars or Streamlit secrets to switch feature source:
            - `SIMILAR_PLAYER_FEATURE_SOURCE_MODE=local|bigquery`
            - `SIMILAR_PLAYER_LOCAL_FEATURE_PATH=artifacts/features/processed_features.parquet`
            - `SIMILAR_PLAYER_BQ_PROJECT=<your-project>`
            - `SIMILAR_PLAYER_BQ_TABLE=feature.v_dashboard_players`
            - `SIMILAR_PLAYER_EMBEDDING_PATH=artifacts/embeddings/player_embedding_table_6d.parquet`
            """
        )


@st.cache_data
def load_data(config: dict):
    df = load_features(
        source_mode=config["mode"],
        local_path=config["local_path"],
        bq_project=config["bq_project"],
        bq_table=config["bq_table"],
    )
    embedding_df = pd.read_parquet(config["embedding_path"])
    return df, embedding_df


@st.cache_resource
def build_matrices(df):
    return build_similarity_matrices(df, FEATURE_COLS)


def validate_feature_columns(df: pd.DataFrame):
    required_columns = ["player_key", "player_name", "position_group", *FEATURE_COLS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_columns)}")


def validate_embedding_columns(df: pd.DataFrame):
    required_columns = ["player_key", "player_name", "position_group"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required embedding columns: {', '.join(missing_columns)}")

    embedding_columns = [column for column in df.columns if column.startswith("embedding_")]
    if not embedding_columns:
        raise ValueError("No embedding columns found in embedding table.")


def build_common_player_options(feature_df: pd.DataFrame, embedding_df: pd.DataFrame) -> pd.DataFrame:
    feature_players = get_player_options(feature_df).copy()
    embedding_keys = set(embedding_df["player_key"].astype(str))

    common_players = feature_players[
        feature_players["player_key"].astype(str).isin(embedding_keys)
    ].copy()

    if common_players.empty:
        raise ValueError("No overlapping players found between feature data and embedding data.")

    return common_players.sort_values("label").reset_index(drop=True)


config = get_feature_source_config()
show_data_source_help(config)

try:
    df, embedding_df = load_data(config)
except Exception as exc:
    st.error("Unable to load recommendation data.")
    st.exception(exc)
    st.stop()

validate_feature_columns(df)
validate_embedding_columns(embedding_df)

df = df.copy()
embedding_df = embedding_df.copy()
df["player_key"] = df["player_key"].astype(str)
embedding_df["player_key"] = embedding_df["player_key"].astype(str)

sim_matrices = build_matrices(df)

player_options = build_common_player_options(df, embedding_df)
selected_label = st.selectbox("Select a player", player_options["label"].tolist())

selected_row = player_options[player_options["label"] == selected_label].iloc[0]
player_id = str(selected_row["player_key"])
player_name = selected_row["player_name"]
player_position = selected_row["position_group"]

st.write(f"**Selected Player:** {player_name}")
st.write(f"**Position:** {player_position}")

top_k = st.slider("Top K", min_value=3, max_value=10, value=DEFAULT_TOP_K)
same_position_only = st.checkbox("Same position only", value=True)

if st.button("Run Recommendation", key="run_button"):
    baseline_result = recommend_players(
        df=df,
        sim_matrices=sim_matrices,
        player_id=player_id,
        top_k=top_k,
        method="cosine",
        same_position_only=same_position_only
    )

    embed_result = recommend_players_embedding(
        embedding_df=embedding_df,
        player_id=player_id,
        top_k=top_k,
        same_position_only=same_position_only
    )

    baseline_result = baseline_result.copy()
    embed_result = embed_result.copy()

    baseline_result["similarity_score"] = baseline_result["similarity_score"].round(3)
    embed_result["similarity_score"] = embed_result["similarity_score"].round(3)

    show_cols = ["player_name", "position_group", "similarity_score"]

    col1, col2 = st.columns([1.2, 1.2])

    with col1:
        st.subheader("Baseline (Cosine)")
        st.dataframe(
            baseline_result[show_cols],
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.subheader("Autoencoder Embedding (6d)")
        st.dataframe(
            embed_result[show_cols],
            use_container_width=True,
            hide_index=True
        )

    summary = compare_models(
        baseline_df=baseline_result,
        embedding_df=embed_result,
        target_position=player_position
    )

    metric_dict = dict(zip(summary["metric"], summary["value"]))

    st.markdown("---")
    st.subheader("Evaluation Summary")

    m1, m2, m3 = st.columns(3)
    m4, m5 = st.columns(2)

    with m1:
        st.metric("Overlap@K", f"{int(metric_dict['overlap_count'])}/{top_k}")

    with m2:
        st.metric("Baseline Purity", f"{metric_dict['baseline_position_purity']:.3f}")

    with m3:
        st.metric("Embedding Purity", f"{metric_dict['embedding_position_purity']:.3f}")

    with m4:
        st.metric("Baseline Avg Sim", f"{metric_dict['baseline_avg_similarity']:.3f}")

    with m5:
        st.metric("Embedding Avg Sim", f"{metric_dict['embedding_avg_similarity']:.3f}")

    baseline_ids = set(baseline_result["player_key"].astype(str))
    embed_ids = set(embed_result["player_key"].astype(str))
    overlap_ids = baseline_ids & embed_ids

    if overlap_ids:
        overlap_names = embed_result[
            embed_result["player_key"].astype(str).isin(overlap_ids)
        ]["player_name"].tolist()
        st.write("**Common recommended players:**", overlap_names)
