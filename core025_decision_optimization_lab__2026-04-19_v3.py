#!/usr/bin/env python3
# BUILD: core025_ultimate_walkforward_v9_corrected_defaults__2026-04-19

import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="Core025 Ultimate v9 Corrected", layout="wide")
st.title("🎯 Core025 Ultimate Walk-Forward v9 - Corrected Defaults")
st.caption("BUILD: core025_ultimate_walkforward_v9_corrected_defaults__2026-04-19 | 40 plays, 10 Top2 daily")

data_file = st.file_uploader("Upload prepared_full_truth_with_stream_stats_v6.csv", type="csv", key="v9_corr")
if not data_file:
    st.stop()

df = pd.read_csv(data_file)

def normalize_win(x):
    if pd.isna(x) or str(x).strip() == "":
        return ""
    s = str(x).strip().replace(" ", "")
    if s in ["25", "0025", "025"]: return "0025"
    if s in ["225", "0225"]: return "0225"
    if s in ["255", "0255"]: return "0255"
    if s.isdigit():
        s = s.zfill(4)
        if s in ["0025", "0225", "0255"]: return s
    return s

if "WinningMember" in df.columns:
    df["TrueMember"] = df["WinningMember"].apply(normalize_win)
else:
    df["TrueMember"] = df.get("TrueMember", pd.Series([""] * len(df))).apply(normalize_win)

st.success(f"Loaded {len(df)} rows | TrueMember counts: {df['TrueMember'].value_counts().to_dict()}")

MEMBERS = ["0025", "0225", "0255"]

# Sliders with your corrected defaults
col1, col2, col3 = st.columns(3)
with col1:
    max_plays_per_day = st.slider("Max Plays per Day", 20, 60, 40, key="plays_v9")
    max_top2_per_day = st.slider("Max Top2 per Day", 0, 15, 10, key="top2_v9")
    min_margin = st.slider("Min Margin for Top2", 0.0, 3.0, 0.8, step=0.1, key="margin_v9")
with col2:
    prune_pct = st.slider("Prune Low-Density %", 0, 60, 35, key="prune_v9")
    seed_boost = st.slider("Seed Boost", 0.0, 5.0, 2.0, key="seed_v9")
with col3:
    trait_weight = st.slider("Trait Weight", 0.0, 4.0, 2.5, key="trait_v9")
    warm_up = st.slider("Warm-up Rows", 20, 100, 30, key="warm_v9")

def heavy_score_row(row, df_full):
    base = {m: 1.0 for m in MEMBERS}
    
    seed_str = str(row.get("seed", "")).strip()
    if seed_str and seed_str != "None":
        if any(d in seed_str for d in "90"):
            base["0255"] += seed_boost
        digit_sum = sum(int(d) for d in seed_str if d.isdigit())
        if digit_sum % 2 == 0:
            base["0025"] += seed_boost * 0.7
        if len(set(seed_str)) <= 2:
            base["0225"] += seed_boost * 0.8
    
    # Heavy trait activation
    trait_cols = [c for c in df_full.columns if any(k in c.lower() for k in 
                 ["pair_has_", "adj_ord_has_", "parity_pattern", "highlow_pattern", 
                  "pair_tokens", "repeat_shape", "palindrome", "consec", "mirror"])]
    
    for col in trait_cols:
        val = str(row.get(col, "")).strip()
        if val and val != "None" and val != "":
            for m in MEMBERS:
                mask = (df_full[col].astype(str).str.strip() == val) & (df_full["TrueMember"] == m)
                total_with_trait = (df_full[col].astype(str).str.strip() == val).sum()
                if total_with_trait > 0:
                    freq = mask.sum() / total_with_trait
                    if freq > 0.45:
                        base[m] += trait_weight * freq
    
    sorted_scores = sorted(base.items(), key=lambda x: x[1], reverse=True)
    top = sorted_scores[0][0]
    second = sorted_scores[1][0]
    margin = sorted_scores[0][1] - sorted_scores[1][1]
    return top, second, margin

if st.button("🚀 Run v9 with Daily Limits"):
    if "hit_density" in df.columns:
        thresh = df["hit_density"].quantile(prune_pct / 100.0)
        df_p = df[df["hit_density"] >= thresh].copy()
        st.info(f"Pruned to {len(df_p)} rows")
    else:
        df_p = df.copy()
    
    results = []
    plays_today = 0
    top2_today = 0
    
    for i in range(warm_up, len(df_p)):
        if plays_today >= max_plays_per_day:
            break
        
        row = df_p.iloc[i]
        top, second, margin = heavy_score_row(row, df)
        
        true_m = str(row.get("TrueMember", "")).strip()
        top1 = 1 if top == true_m else 0
        needed = 1 if top1 == 0 and second == true_m else 0
        
        if needed:
            if top2_today >= max_top2_per_day:
                needed = 0
                miss = 1
            else:
                top2_today += 1
                miss = 0
        else:
            miss = 1 if top1 == 0 else 0
        
        waste = 1 if needed == 1 and margin < min_margin else 0
        
        results.append({
            "idx": i,
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
        
        plays_today += 1
    
    res_df = pd.DataFrame(results)
    total = len(res_df)
    t1 = res_df["Top1_Correct"].sum()
    nt2 = res_df["Needed_Top2"].sum()
    waste = res_df["Waste_Top2"].sum()
    miss = res_df["Miss"].sum()
    capture = (t1 + nt2) / total * 100 if total > 0 else 0
    obj = (t1 * 3) + (nt2 * 2) - (waste * 1.2) - (miss * 2.5)
    
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
        st.metric("Plays/Win", f"{total / (t1 + nt2) if (t1 + nt2) > 0 else total:.2f}")
    
    st.dataframe(res_df)
    
    csv = res_df.to_csv(index=False)
    st.download_button(
        "📥 Download walkforward_results_v9.csv",
        data=csv,
        file_name="walkforward_results_v9.csv",
        mime="text/csv",
        key="v9_corrected"
    )

st.caption("Defaults corrected: Max Plays = 40, Max Top2 per Day = 10. Move sliders as needed and run.")
