import pandas as pd
from collections import Counter, defaultdict
import ast
import json
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


with open(f"{PROJECT_ROOT_DIR}/config/position_mapping.json","r") as f:
    position_mapping = json.load(f)
with open(f'{PROJECT_ROOT_DIR}/config/position_mapping_level_1.json', 'r') as file:
    position_mapping_level_1 = json.load(file)
reverse_mapping = {pos: level for level, positions in position_mapping_level_1.items() for pos in positions}


positions_by_role = {
    "Goalkeeper": [
        "GK",
        "Goalkeeper"
    ],
    "Defender": [
        "RB", "RCB", "CB", "LCB", "LB", "RWB", "LWB",
        "Right Back", "Right Center Back", "Center Back", "Left Center Back",
        "Left Back", "Right Wing Back", "Left Wing Back"
    ],
    "Midfielder": [
        "RDM", "CDM", "LDM", "RM", "RCM", "CM", "LCM", "LM", "RW", 
        "RAM", "CAM", "LAM",
        "Right Defensive Midfield", "Center Defensive Midfield", "Left Defensive Midfield",
        "Right Midfield", "Right Center Midfield", "Center Midfield", "Left Center Midfield", "Left Midfield",
        "Right Attacking Midfield", "Center Attacking Midfield", "Left Attacking Midfield"
    ],
    "Forward": [
        "LW", "RCF", "ST", "LCF", "SS",
        "Left Wing", "Right Wing", "Right Center Forward",
        "Striker", "Left Center Forward", "Secondary Striker", "Center Forward"
    ]
}

def get_player_role(row, positions_by_role_map=positions_by_role):
    """
    Return the most representative role from `positions_played` with tie-breaking:
      1) Prefer the last occurrence among tied roles that belong to the player's global position.
      2) Otherwise, prefer the last occurrence among tied roles.
    Assumes `row["positions_played"]` is a list or a stringified list.
    """
    played = row.get("positions_played", None)
    global_position = row.get("new_position_level_0", None)

    # Normalize to list
    if isinstance(played, str):
        try:
            played = ast.literal_eval(played)
        except Exception:
            # last-resort fallback if it's a comma-separated string
            played = [p.strip() for p in played.split(",") if p.strip()]

    if not played:  # None, empty list, etc.
        return None

    # Count occurrences
    counts = Counter(played)
    allowed = set(positions_by_role.get(global_position, [])) if global_position else set()

    if allowed:
        # restrict to allowed positions
        counts_allowed = {p: c for p, c in counts.items() if p in allowed}
        if counts_allowed:
            max_allowed = max(counts_allowed.values())
            # tie-break: last occurrence among the top-allowed
            for pos in reversed(played):
                if counts_allowed.get(pos) == max_allowed:
                    return pos
            # (shouldn’t reach here)
    
    # Fallback: global most frequent, tie-break by last occurrence
    max_count = max(counts.values())
    for pos in reversed(played):
        if counts.get(pos) == max_count:
            return pos


def decide_global_position(row, mapping=position_mapping):
    """
    Maps a list of exact positions -> global positions.
    Returns (chosen_global, counts_by_global, mapped_sequence).
    Tie-break rule: if multiple globals have the same max count,
    pick the last played global (i.e., last in the mapped sequence).
    """
    positions_played = row.get("positions_played", None)
    # Normalize to list
    if isinstance(positions_played, str):
        try:
            positions_played = ast.literal_eval(positions_played)
        except Exception:
            # last-resort fallback if it's a comma-separated string
            positions_played = [p.strip() for p in positions_played.split(",") if p.strip()]

    if not positions_played:  # None, empty list, etc.
        return None

    mapped = [mapping.get(p) for p in positions_played if mapping.get(p) is not None]
    if not mapped:
        return None# , Counter(), []

    counts = Counter(mapped)
    max_count = max(counts.values())
    tied = {g for g, c in counts.items() if c == max_count}

    # Tie-break: scan from the end and pick the first that belongs to the tied set
    for g in reversed(mapped):
        if g in tied:
            return g #, counts, mapped

    # Fallback (shouldn’t happen)
    return mapped[-1]

if __name__ == "__main__":
    # === load standard stats ===
    df = pd.read_csv(f"{PROJECT_ROOT_DIR}/data/new_approach/standard_stats_all_test.csv")

    # === level 0 ===
    df["new_position_level_0"] = df.apply(decide_global_position, axis=1)
    df["new_position_level_2"] = df.apply(get_player_role, axis=1)
    df["new_position_level_1"] = df["new_position_level_2"].map(reverse_mapping)
    df = df.drop(columns=["position_level_0", "position_level_1", "position_level_2"])
    df = df.rename(columns={
        "new_position_level_0": "position_level_0",
        "new_position_level_1": "position_level_1",
        "new_position_level_2": "position_level_2"
    })
    df.to_csv(f"{PROJECT_ROOT_DIR}/data/new_approach/standard_stats_all_final.csv", index=False)
