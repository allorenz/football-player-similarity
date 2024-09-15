import streamlit as st
import numpy as np


"""
To run: 
streamlit run test-app.py
"""

# UI for user preferences
st.title("Player Recommender System")

# Sliders for attribute weighting
speed_weight = st.slider('Speed Weight', 0.0, 1.0, 0.5)
crossing_weight = st.slider('Crossing Weight', 0.0, 1.0, 0.3)
passing_accuracy_weight = st.slider('Passing Accuracy Weight', 0.0, 1.0, 0.2)

# Display selected weights
st.write(f"Selected Weights: Speed: {speed_weight}, Crossing: {crossing_weight}, Passing Accuracy: {passing_accuracy_weight}")

# Function that uses the weighted similarity (Euclidean or Cosine)
def calculate_similarity(player_stats, query_stats, weights):
    # Apply weights and calculate Euclidean distance as an example
    weighted_diff = weights * (player_stats - query_stats)**2
    return np.sqrt(np.sum(weighted_diff))

# Placeholder for future implementation to use these weights to recommend players
