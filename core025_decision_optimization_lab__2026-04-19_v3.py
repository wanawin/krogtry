# BUILD: core025_northern_lights__v40_verified_top3_downloads

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# -----------------------------
# CONSTANTS
# -----------------------------
MEMBERS = ["0025", "0225", "0255"]

# -----------------------------
# FILE LOAD
# -----------------------------
st.sidebar.header("Upload Data")
file = st.sidebar.file_uploader("Upload Lab Per Event CSV", type=["csv"])

if file is None:
    st.warning("Upload your v39 lab_per_event file to continue.")
    st.stop()

df = pd.read_csv(file)

# -----------------------------
# BASIC VALIDATION
# -----------------------------
required_cols = [
    "StreamKey", "SingleRow", "RowPercentile",
    "PredictedMember", "Top2_pred", "TrueMember",
    "Top1_Correct", "Needed_Top2", "Miss"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# -----------------------------
# THIRD MEMBER
# -----------------------------
def get_third(row):
    return [m for m in MEMBERS if m not in [row["PredictedMember"], row["Top2_pred"]]][0]

df["ThirdMember"] = df.apply(get_third, axis=1)

# -----------------------------
# TOP3 RESCUE LOGIC (MEASURED)
# -----------------------------
danger_rows = {2, 14, 30, 50}
