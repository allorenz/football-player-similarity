import streamlit as st
import pandas as pd
from recommendation import Recommender


rec = Recommender()

# --- Streamlit App ---
st.title("Player Recommendation System")

# Searchable select box for player names
selected_player = st.selectbox("Search and select a player:", rec.df_standard_stats["player"].to_list())

# When a player is selected, call recommend and show the result
if selected_player:
    output_df = rec.recommend(selected_player)
    st.subheader(f"Recommendations for {selected_player}:")
    st.dataframe(output_df)
