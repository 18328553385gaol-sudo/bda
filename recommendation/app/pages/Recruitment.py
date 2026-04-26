import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from config import DEFAULT_TOP_K
from data_loader import load_local_features
from preprocess import build_dimension_scores
from recommender import recruit_players

st.title("📋 Recruitment Ranking")

st.caption("Rank players by weighted scouting dimensions within a target position.")

@st.cache_data
def load_rank_data():
    df = load_local_features()
    df_rank = build_dimension_scores(df)
    return df_rank

df_rank = load_rank_data()

position_options = sorted(df_rank["position_group"].dropna().unique().tolist())
target_position = st.selectbox("Target Position", position_options)

st.markdown("### Dimension Weights")

col1, col2 = st.columns(2)

with col1:
    attacking_weight = st.slider("Attacking", 0.0, 1.0, 0.3, 0.05)
    progression_weight = st.slider("Progression", 0.0, 1.0, 0.3, 0.05)

with col2:
    defensive_weight = st.slider("Defensive", 0.0, 1.0, 0.3, 0.05)
    spatial_weight = st.slider("Spatial", 0.0, 1.0, 0.1, 0.05)

top_k = st.slider("Top K", 3, 10, DEFAULT_TOP_K)

weight_sum = attacking_weight + progression_weight + defensive_weight + spatial_weight
st.info(f"Current weight sum: {weight_sum:.2f}")

if st.button("Run Recruitment Ranking"):
    if weight_sum == 0:
        st.error("Weight sum cannot be 0. Please assign at least one non-zero weight.")
    else:
        # 自动归一化，保证总和为 1
        attacking_weight_norm = attacking_weight / weight_sum
        progression_weight_norm = progression_weight / weight_sum
        defensive_weight_norm = defensive_weight / weight_sum
        spatial_weight_norm = spatial_weight / weight_sum

        result = recruit_players(
            df=df_rank,
            target_position=target_position,
            top_k=top_k,
            attacking_weight=attacking_weight_norm,
            progression_weight=progression_weight_norm,
            defensive_weight=defensive_weight_norm,
            spatial_weight=spatial_weight_norm
        )

        show_cols = ["player_key", "player_name", "position_group", "recruitment_score"]

        st.subheader("Top Candidates")
        st.dataframe(result[show_cols], use_container_width=True)

        st.markdown("### Normalized Weights Used")
        st.write({
            "attacking": round(attacking_weight_norm, 3),
            "progression": round(progression_weight_norm, 3),
            "defensive": round(defensive_weight_norm, 3),
            "spatial": round(spatial_weight_norm, 3),
        })