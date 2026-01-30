import streamlit as st
import pandas as pd
import plotly.express as px
from feature_selection import filter_df
from sklearn.preprocessing import MinMaxScaler

# --- Sample Player Data (with player_id as index) ---
data = {
    "player_id": [101, 102, 103, 104],
    "Player": ["Alice", "Bob", "Charlie", "Diana"],
    # Passing
    "Short Pass": [85, 70, 90, 65],
    "Long Pass": [80, 65, 88, 70],
    "Crossing": [78, 72, 85, 68],
    # Possession
    "Ball Control": [90, 75, 88, 80],
    "Dribbling": [85, 70, 90, 75],
    "Vision": [88, 68, 85, 72],
    # Shooting
    "Finishing": [80, 85, 75, 70],
    "Shot Power": [82, 87, 77, 74],
    "Long Shots": [78, 83, 74, 76],
    # Defending
    "Tackling": [70, 88, 60, 85],
    "Interceptions": [75, 90, 65, 82],
    "Marking": [72, 89, 62, 80],
}
df = pd.DataFrame(data).set_index("player_id")

def load_data():
    scaler = MinMaxScaler()
    df_standard_stats = pd.read_csv(
            "../../data/new_approach/standard_stats_all_final.csv", dtype={"player_id": "int32"}
    ).set_index("player_id", drop=True)
    # vars
    dimensions = ["defending","possession", "passing", "shooting"] # "goal_keeping"]
    df_standard_stats = pd.read_csv("../../data/new_approach/standard_stats_all_final.csv",dtype={"player_id":"int32"}) # load_standard_stats(unique_index=True)

    # Merge all dimensions
    df = df_standard_stats[["player_id","player", "match_played", "minutes_played"]].copy()
    for dim in dimensions:
        # load
        df_dimension = pd.read_csv(f"../../data/new_approach/{dim}_ex.csv",dtype={"player_id":"int32"})

        # merge and update base df
        df = pd.merge(
            left=df,
            right=df_dimension.loc[:, df_dimension.columns != "player"],
            left_on="player_id", 
            right_on="player_id",
            how="left"
        )
    df_filtered = filter_df(df, match_played=2, minutes_played=90)
    df_filtered = df_filtered.set_index("player_id", drop=True)
    numeric_cols = df_filtered.select_dtypes(include="number").columns
    df_filtered[numeric_cols] = scaler.fit_transform(df_filtered[numeric_cols])

    return df_filtered

df = load_data()

# --- Skill Categories ---
categories = {
    "Passing": ["passes_completed", "passes_into_box","passes_short_completed","passes_medium_total","passes_medium_completed","passes_goal_assist","passes_shot_assist"],
    "Possession": ["touches_total", "dribbling_total", "progressive_carry_total", "progressive_carry_distance","dispossessed_total"],
    "Shooting": ["shots_total", "shots_on_target", "goals","goals_with_first_touch","xG","big_chances_led_to_goal"],
    "Defending": ["ball_recovery_total", "pressure_on_opponent", "counterpressing_total","block_total"]
}

# --- Sidebar Player Selection ---
st.sidebar.header("Compare Players")

# Create dictionary for display name → player_id
player_name_map = df["player"].to_dict()           # {player_id: name}
player_options = {v: k for k, v in player_name_map.items()}  # {name: player_id}

# Set default player names (case-insensitive match)
def get_player_name(target):
    for name in player_options.keys():
        if name.lower() == target.lower():
            return name
    return list(player_options.keys())[0]  # fallback

default_a = get_player_name("Thomas Müller")
default_b = get_player_name("Cristiano Ronaldo dos Santos Aveiro")

player_a_name = st.sidebar.selectbox("Choose Player A", options=player_options.keys(), key="player_a", index=list(player_options.keys()).index(default_a))
player_b_name = st.sidebar.selectbox("Choose Player B", options=player_options.keys(), key="player_b", index=list(player_options.keys()).index(default_b))

player_a_id = player_options[player_a_name]
player_b_id = player_options[player_b_name]

# --- Main UI ---
st.title("Detailed Player Comparison")
st.set_page_config(layout="wide")

if player_a_id == player_b_id:
    st.warning("Please select two different players.")
else:
    selected_df = df.loc[[player_a_id, player_b_id]].copy()

    # 2x2 Radar Chart Layout
    chart_sections = list(categories.items())
    for i in range(0, len(chart_sections), 2):
        col1, col2 = st.columns([4, 4])

        for col, (section, attributes) in zip((col1, col2), chart_sections[i:i+2]):
            melted = selected_df[["player"] + attributes].reset_index().melt(
                id_vars=["player_id", "player"], var_name="Attribute", value_name="Value"
            )
            fig = px.line_polar(
                melted,
                r="Value",
                theta="Attribute",
                color="player",
                line_close=True,
                template="plotly_dark"
            )
            fig.update_traces(fill="toself")

            col.subheader(section)
            col.plotly_chart(fig, use_container_width=True)
