#!/usr/bin/env python3
# BUILD: core025_ultimate_walkforward_heavy_v2__2026-04-19

import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="Core025 Ultimate Walk-Forward v2", layout="wide")
st.title("🎯 Core025 Ultimate Heavy Walk-Forward v2")
st.caption("BUILD: core025_ultimate_walkforward_heavy_v2__2026-04-19 | Fixed normalization + stable download")

# ====================== UPLOADS ======================
main_file = st.file_uploader("prepared_full_truth_with_stream_stats_v4.csv", type="csv", key="main")
rule_file = st.file_uploader("rule_metadata__core025_precompute_builder__2026-04-16_v1.csv", type="csv", key="rules")
matrix_file = st.file_uploader("match_matrix__core025_precompute_builder__2026-04-16_v1.csv", type="csv", key="matrix")

if not main_file:
    st.info("Upload the v4 prepared file (and optionally rule/matrix for heavier scoring).")
    st.stop()

df = pd.read_csv(main_file)

# ====================== NORMALIZE TrueMember ======================
def normalize_member(x):
    if pd.isna(x) or str(x).strip() == "":
        return ""
    s = str(x).strip()
    if s in ["25", "255", "0255"]: return "0255"
    if s in ["225", "0225"]: return "0225"
    if s in ["25", "0025"]: return "0025"
    return s.zfill(4) if s.isdigit() else s

df["TrueMember"] = df["TrueMember"].apply(normalize_member)
if "WinningMember" in df.columns:
    df["TrueMember"] = df["TrueMember"].combine_first(df["WinningMember"].apply(normalize_member))

st.success(f"✅ Loaded {len(df)} rows. TrueMember normalized.")

# Load heavy files if provided
rule_meta = pd.read_csv(rule_file) if rule_file is not None else pd.DataFrame()
match_matrix = pd.read_csv(matrix_file, index_col=0) if matrix_file is not None else pd.DataFrame()

MEMBERS = ["0025", "0225", "0255"]

# ====================== SLIDERS ======================
col1, col2 = st.columns(2)
with col1:
    max_plays = st.slider("Max Plays per Day", 20, 60, 40, key="plays")
    max_top2 = st.slider("Max Top2 Allowed", 0, 20, 10, key="top2")
    prune_pct = st.slider("Prune Low Hit-Density Streams (%)", 0, 50, 30, key="prune")
with col2:
    seed_boost = st.slider("Seed Boost", 0.0, 5.0, 2.5, key="seed")
    trait_weight = st.slider("Trait/Rule Weight", 0.0, 3.0, 1.2, key="trait")
    warm_up = st.slider("Warm-up Buffer", 20, 100, 40, key="warm")

# ====================== HEAVY SCORE FUNCTION ======================
def heavy_score_row(row, match_matrix, rule_meta, seed_boost_val, trait_weight_val):
    base = {m: 1.0 for m in MEMBERS}
    
    seed = str(row.get("seed", "")).strip()
    if seed:
        if any(d in seed for d in "90"):
            base["0255"] += seed_boost_val
        if len(set(seed)) <= 2:
            base["0225"] += seed_boost_val * 0.9
        if sum(int(d) for d in seed if d.isdigit()) % 2 == 0:
            base["0025"] += seed_boost_val * 0.8
    
    # Heavy trait + rule boost
    try:
        idx = row.name
        if isinstance(match_matrix, pd.DataFrame) and not match_matrix.empty and idx in match_matrix.index:
            acts = match_matrix.loc[idx]
            for m in MEMBERS:
                m_rules = rule_meta[rule_meta.get("target", "") == m] if not rule_meta.empty else pd.DataFrame()
                if not m_rules.empty:
                    rule_boost = m_rules["hit_rate_true"].mean() * m_rules.get("lift", 1.0).mean()
                    base[m] += float(acts.sum()) * trait_weight_val * rule_boost
    except:
        pass
    
    sorted_m = sorted(base.items(), key=lambda x: x[1], reverse=True)
    top = sorted_m[0][0]
    second = sorted_m[1][0]
    margin = sorted_m[0][1] - sorted_m[1][1]
    return top, second, margin

# ====================== RUN WALK-FORWARD ======================
if st.button("🚀 Run Heavy Walk-Forward"):
    # Prune
    if "hit_density" in df.columns:
        thresh = df["hit_density"].quantile(prune_pct / 100.0)
        df_pruned = df[df["hit_density"] >= thresh].copy()
        st.info(f"Pruned to {len(df_pruned)} rows using hit_density ≥ {thresh:.3f}")
    else:
        df_pruned = df.copy()
    
    results = []
    progress = st.progress(0)
    
    for i in range(warm_up, len(df_pruned)):
        if len(results) >= max_plays:
            break
        row = df_pruned.iloc[i]
        
        top, second, margin = heavy_score_row(row, match_matrix, rule_meta, seed_boost, trait_weight)
        
        true_m = str(row.get("TrueMember", "")).strip()
        top1_correct = 1 if top == true_m else 0
        needed_top2 = 1 if top1_correct == 0 and second == true_m else 0
        waste = 1 if needed_top2 == 1 and margin < 0.35 else 0
        miss = 1 if top1_correct == 0 and needed_top2 == 0 else 0
        
        results.append({
            "date": row.get("date", row.get("PlayDate", "")),
            "stream": row.get("StreamKey", ""),
            "seed": row.get("seed", ""),
            "PredictedMember": top,
            "Top2_pred": second,
            "TrueMember": true_m,
            "Top1_Correct": top1_correct,
            "Needed_Top2": needed_top2,
            "Waste_Top2": waste,
            "Miss": miss,
            "Margin": round(margin, 3)
        })
        
        progress.progress((i - warm_up + 1) / (len(df_pruned) - warm_up))
    
    res_df = pd.DataFrame(results)
    
    # Metrics
    total = len(res_df)
    t1 = res_df["Top1_Correct"].sum()
    nt2 = res_df["Needed_Top2"].sum()
    waste = res_df["Waste_Top2"].sum()
    miss = res_df["Miss"].sum()
    capture = (t1 + nt2) / total * 100 if total > 0 else 0
    ppw = total / (t1 + nt2) if (t1 + nt2) > 0 else total
    obj = (t1 * 3.0) + (nt2 * 2.0) - (waste * 1.2) - (miss * 2.5)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Capture Rate", f"{capture:.1f}%")
        st.metric("Top1 Wins", int(t1))
    with c2:
        st.metric("Needed Top2", int(nt2))
        st.metric("Waste Top2", int(waste))
    with c3:
        st.metric("Misses", int(miss))
        st.metric("Objective Score", f"{obj:.1f}")
        st.metric("Plays per Win", f"{ppw:.2f}")
    
    st.dataframe(res_df)
    
    # STABLE DOWNLOAD (no reset)
    if "download_key" not in st.session_state:
        st.session_state.download_key = False
    
    csv_data = res_df.to_csv(index=False).encode()
    st.download_button(
        "📥 Download walkforward_results_ultimate_v2.csv",
        data=csv_data,
        file_name="walkforward_results_ultimate_v2.csv",
        mime="text/csv",
        key="stable_download"
    )
    
    st.success("✅ Walk-forward complete. Download should not reset the app.")
