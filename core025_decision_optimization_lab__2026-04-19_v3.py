#!/usr/bin/env python3
# BUILD: core025_ultimate_walkforward_v7__2026-04-19

import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="Core025 Ultimate v7", layout="wide")
st.title("🎯 Core025 Ultimate Walk-Forward v7")
st.caption("BUILD: core025_ultimate_walkforward_v7__2026-04-19 | Direct WinningMember normalization")

data_file = st.file_uploader("Upload prepared_full_truth_with_stream_stats_v6.csv (or any prepared file)", type="csv", key="data")
if not data_file:
    st.stop()

df = pd.read_csv(data_file)

# Strong normalization from the column that actually has data
def normalize_win(x):
    if pd.isna(x) or str(x).strip() == "":
        return ""
    s = str(x).strip().replace(" ", "")
    if s in ["25", "0025", "025"]: return "0025"
    if s in ["225", "0225"]: return "0225"
    if s in ["255", "0255"]: return "0255"
    if s.isdigit():
        s = s.zfill(4)
        if s == "0025": return "0025"
        if s == "0225": return "0225"
        if s == "0255": return "0255"
    return s

# Use WinningMember if present, otherwise TrueMember
if "WinningMember" in df.columns:
    df["TrueMember"] = df["WinningMember"].apply(normalize_win)
elif "TrueMember" in df.columns:
    df["TrueMember"] = df["TrueMember"].apply(normalize_win)
else:
    df["TrueMember"] = ""

# Debug
st.success(f"Loaded {len(df)} rows.")
st.subheader("🔍 TrueMember Debug (Raw → Normalized)")
if "WinningMember" in df.columns:
    st.write("Raw WinningMember counts:", df["WinningMember"].value_counts().to_dict())
st.write("Normalized TrueMember counts:", df["TrueMember"].value_counts().to_dict())

MEMBERS = ["0025", "0225", "0255"]

# Sliders
col1, col2 = st.columns(2)
with col1:
    max_plays = st.slider("Max Plays per Day", 20, 60, 40)
    max_top2 = st.slider("Max Top2 Allowed", 0, 20, 10)
    prune_pct = st.slider("Prune Low-Density %", 0, 50, 25)
with col2:
    seed_boost = st.slider("Seed Boost", 0.0, 5.0, 2.5)
    trait_weight = st.slider("Trait Weight", 0.0, 3.0, 1.2)
    warm_up = st.slider("Warm-up Rows", 20, 100, 40)

def score_row(row):
    base = {m: 1.0 for m in MEMBERS}
    seed = str(row.get("seed", "")).strip()
    if seed:
        if any(d in seed for d in "90"):
            base["0255"] += seed_boost
        if len(set(seed)) <= 2:
            base["0225"] += seed_boost * 0.9
        digit_sum = sum(int(d) for d in seed if d.isdigit())
        if digit_sum % 2 == 0:
            base["0025"] += seed_boost * 0.8
    sorted_scores = sorted(base.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[0][0], sorted_scores[1][0], sorted_scores[0][1] - sorted_scores[1][1]

if st.button("🚀 Run Walk-Forward v7"):
    if "hit_density" in df.columns:
        thresh = df["hit_density"].quantile(prune_pct / 100.0)
        df_p = df[df["hit_density"] >= thresh].copy()
        st.info(f"Pruned to {len(df_p)} rows")
    else:
        df_p = df.copy()
    
    results = []
    for i in range(warm_up, len(df_p)):
        if len(results) >= max_plays:
            break
        row = df_p.iloc[i]
        top, second, margin = score_row(row)
        
        true_m = str(row.get("TrueMember", "")).strip()
        top1 = 1 if top == true_m else 0
        needed = 1 if top1 == 0 and second == true_m else 0
        waste = 1 if needed == 1 and margin < 0.35 else 0
        miss = 1 if top1 == 0 and needed == 0 else 0
        
        results.append({
            "date": row.get("date", row.get("PlayDate", "")),
            "stream": row.get("StreamKey", ""),
            "seed": row.get("seed", ""),
            "PredictedMember": top,
            "Top2_pred": second,
            "TrueMember": true_m,
            "Top1_Correct": top1,
            "Needed_Top2": needed,
            "Waste_Top2": waste,
            "Miss": miss,
            "Margin": round(margin, 3)
        })
    
    res_df = pd.DataFrame(results)
    
    total = len(res_df)
    t1 = res_df["Top1_Correct"].sum()
    nt2 = res_df["Needed_Top2"].sum()
    waste = res_df["Waste_Top2"].sum()
    miss = res_df["Miss"].sum()
    capture = (t1 + nt2) / total * 100 if total > 0 else 0
    obj = (t1*3) + (nt2*2) - (waste*1.2) - (miss*2.5)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Capture Rate", f"{capture:.1f}%")
        st.metric("Top1 Wins", int(t1))
    with c2:
        st.metric("Needed Top2", int(nt2))
        st.metric("Waste Top2", int(waste))
    with c3:
        st.metric("Misses", int(miss))
        st.metric("Objective", f"{obj:.1f}")
        st.metric("Plays/Win", f"{total/(t1+nt2) if (t1+nt2)>0 else total:.2f}")
    
    st.dataframe(res_df)
    
    csv = res_df.to_csv(index=False)
    st.download_button(
        "📥 Download walkforward_results_v7.csv",
        data=csv,
        file_name="walkforward_results_v7.csv",
        mime="text/csv",
        key="v7_stable"
    )
    
    st.success("Run complete.")

st.caption("All data real. Normalization now uses the WinningMember column that actually has values.")
