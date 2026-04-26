import streamlit as st

st.set_page_config(
    page_title="Smart Scouting Recommendation",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Smart Scouting Recommendation System")
st.markdown("""
这个应用展示了两类推荐能力：

1. **Similar Player Recommendation**
   - Baseline: feature-based similarity
   - Learned model: autoencoder embedding similarity

2. **Recruitment Ranking**
   - 根据不同维度权重筛选候选球员
""")

st.info("请从左侧页面进入：Similar Player 或 Recruitment")