import streamlit as st
import pandas as pd
from recommendation import Recommender

# Set full-width layout
st.set_page_config(layout="wide")

rec = Recommender()

# --- Streamlit App ---
st.title("Player Recommendation System (Top 5 Leagues 2015/16)")

# Sidebar
st.sidebar.header("Select Player")

# Player selection
selected_player = st.sidebar.selectbox(
    "Search and select a player:",
    rec.df_standard_stats["player"].to_list(),
    index=rec.df_standard_stats["player"].to_list().index("Thomas Müller") 
)

# Define dimensions
dimension_options = ["defending", "passing", "possession", "shooting"]

# --- Initialize session state for sliders ---
for dim in dimension_options:
    slider_key = f"{dim}_weight"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = 0.5

# --- Reset Button ---
if st.sidebar.button("Reset"):
    for dim in dimension_options:
        st.session_state[f"{dim}_weight"] = 0.5

# Select dimensions and assign weights
st.sidebar.markdown("### Select dimensions and assign weights:")
selected_dimensions = []
dimension_weights_list = []

for dim in dimension_options:
    checked = st.sidebar.checkbox(dim.capitalize(), value=True)
    if checked:
        selected_dimensions.append(dim)
        slider_key = f"{dim}_weight"
        weight = st.sidebar.slider(
            f"{dim.capitalize()} weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state[slider_key],
            step=0.05,
            key=slider_key  # critical: use the same key as session_state
        )
        dimension_weights_list.append(weight)

# Main content
if selected_player and selected_dimensions:
    output_df = rec.recommend(selected_player, selected_dimensions, weights=dimension_weights_list)
    st.subheader(f"Recommendations for {selected_player} based on weighted dimensions:")
    st.dataframe(output_df, use_container_width=True)
    
elif selected_player and not selected_dimensions:
    st.warning("Please select at least one dimension to generate recommendations.")