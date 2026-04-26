import streamlit as st

st.set_page_config(
    page_title="Smart Scouting Recommendation",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Smart Scouting Recommendation System")
st.markdown("""
The app is showing：

1. **Similar Player Recommendation**
   - Baseline: feature-based similarity
   - Learned model: autoencoder embedding similarity

2. **Recruitment Ranking**
   - Use different dimention to search player
""")

st.info("Please Enter for the Left Bar：Similar Player or Recruitment")
