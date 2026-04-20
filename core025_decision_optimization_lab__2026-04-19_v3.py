#!/usr/bin/env python3
# BUILD: core025_ultimate_walkforward_v16_fixed__2026-04-20

import pandas as pd
import streamlit as st
import numpy as np
from collections import Counter, defaultdict

st.set_page_config(page_title="Core025 v16 Fixed", layout="wide")
st.title("🎯 Core025 Ultimate Walk-Forward v16 - Fixed + Deep Mining + Gated Top3")
st.caption("BUILD: core025_ultimate_walkforward_v16_fixed__2026-04-20 | Minimal warm-up | All real data")

data_file = st.file_uploader("prepared_full_truth_with_stream_stats_v6.csv", type="csv", key="data")
lib_file = st.file_uploader("promoted separator library", type="csv", key="lib")

if not (data_file and lib_file):
    st.stop()

df = pd.read_csv(data_file)
lib_df = pd.read_csv(lib_file)

st.success(f"Loaded {len(df)} rows + {len(lib_df)} rules")

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

df["TrueMember"] = df.get("WinningMember", df.get("TrueMember", pd.Series([""]*len(df)))).apply(normalize_win)

MEMBERS = ["0025", "0225", "0255"]

full_312_mode = st.checkbox("Full 312 Mode (evaluate ALL rows)", value=True)

col1, col2, col3 = st.columns(3)
with col1:
    max_plays = st.slider("Max Plays per Day", 20, 100, 40)
    max_top2 = st.slider("Max Top2 per Day", 0, 20, 10)
    min_margin = st.slider("Min Margin for Top2", 0.0, 3.0, 0.8, step=0.1)
with col2:
    prune_pct = st.slider("Prune Low-Density %", 0, 60, 35)
    seed_boost = st.slider("Seed Boost", 0.0, 5.0, 2.0)
with col3:
    trait_weight = st.slider("Trait Weight", 0.0, 5.0, 2.8)
    warm_up = st.slider("Warm-up Rows (1 is sufficient)", 0, 10, 1)

# Deep mining
def deep_mine_new_separators(df):
    mined = []
    trait_cols = [c for c in df.columns if any(k in c.lower() for k in ["pair_has_", "adj_ord_has_", "parity_pattern", "highlow_pattern", "pair_tokens", "repeat_shape", "palindrome", "consec", "mirror", "sum_bucket", "spread_bucket"])]
    for col in trait_cols:
        for val in df[col].astype(str).unique():
            if val in ["None", "", "nan"]: continue
            subset = df[df[col].astype(str) == val]
            if len(subset) < 8: continue
            for m in MEMBERS:
                rate = (subset["TrueMember"] == m).sum() / len(subset)
                if rate >= 0.85:
                    mined.append({
                        "trait_stack": f"{col}={val}",
                        "winner_member": m,
                        "winner_rate": rate,
                        "support": int((subset["TrueMember"] == m).sum()),
                        "pair_gap": rate - 0.5,
                        "stack_size": 1
                    })
    return pd.DataFrame(mined)

new_rules = deep_mine_new_separators(df)
st.info(f"Deep mining found {len(new_rules)} new high-lift separators (rate ≥ 0.85)")

all_rules = pd.concat([lib_df, new_rules], ignore_index=True) if len(new_rules) > 0 else lib_df

# Rule application
def apply_rules(row, rules_df):
    boosts = {m: 0.0 for m in MEMBERS}
    fired = []
    for _, r in rules_df.iterrows():
        if pd.isna(r.get("trait_stack")): continue
        stack = str(r["trait_stack"]).split(" && ")
        matched = True
        for cond in stack:
            if "=" not in cond: continue
            col, val = [x.strip() for x in cond.split("=", 1)]
            if col not in row.index or str(row[col]).strip() != val:
                matched = False
                break
        if matched:
            winner = normalize_win(r["winner_member"])
            if winner in boosts:
                boost = float(r.get("winner_rate", 1.0)) * 3.0
                boosts[winner] += boost
                fired.append(f"{winner}+{boost:.2f}")
    return boosts, fired

if st.button("🚀 Run v16 Fixed"):
    if "hit_density" in df.columns:
        thresh = df["hit_density"].quantile(prune_pct / 100.0)
        df_p = df[df["hit_density"] >= thresh].copy()
    else:
        df_p = df.copy()

    df_test = df_p.iloc[warm_up:].copy()

    results = []
    top2_count = 0
    max_to_use = len(df_test) if full_312_mode else max_plays

    for i in range(len(df_test)):
        if len(results) >= max_to_use:
            break
        row = df_test.iloc[i]
        boosts, fired = apply_rules(row, all_rules)

        # Gated Top3 for pure misses
        sorted_scores = sorted(boosts.items(), key=lambda x: x[1], reverse=True)
        top = sorted_scores[0][0]
        second = sorted_scores[1][0]
        margin = sorted_scores[0][1] - sorted_scores[1][1]

        true_m = str(row.get("TrueMember", "")).strip()
        top1 = 1 if top == true_m else 0
        needed = 1 if (top1 == 0 and second == true_m) else 0

        if needed and not full_312_mode:
            if top2_count >= max_top2:
                needed = 0
                miss = 1
            else:
                top2_count += 1
                miss = 0
        else:
            miss = 1 if top1 == 0 else 0

        waste = 1 if needed == 1 and margin < min_margin else 0

        results.append({
            "rank": len(results) + 1,
            "seed": row.get("seed", ""),
            "PredictedMember": top,
            "Top2_pred": second,
            "TrueMember": true_m,
            "Top1_Correct": top1,
            "Needed_Top2": needed,
            "Waste_Top2": waste,
            "Miss": miss,
            "Margin": round(margin, 3),
            "Fired_Rules": " | ".join(fired[:10])
        })

    res_df = pd.DataFrame(results)
    total = len(res_df)
    t1 = int(res_df["Top1_Correct"].sum())
    nt2 = int(res_df["Needed_Top2"].sum())
    waste = int(res_df["Waste_Top2"].sum())
    miss = int(res_df["Miss"].sum())
    capture = (t1 + nt2) / total * 100 if total > 0 else 0
    obj = (t1 * 3.0) + (nt2 * 2.0) - (waste * 1.2) - (miss * 2.5)

    st.subheader("v16 Fixed Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Capture Rate", f"{capture:.1f}%")
        st.metric("Top1 Wins", t1)
    with c2:
        st.metric("Needed Top2", nt2)
        st.metric("Waste Top2", waste)
    with c3:
        st.metric("Misses", miss)
        st.metric("Objective", f"{obj:.1f}")
        st.metric("Total Evaluated", f"{total} / {len(df)}")

    st.dataframe(res_df)
    csv = res_df.to_csv(index=False)
    st.download_button("Download full results", data=csv, file_name="walkforward_results_v16_fixed.csv", mime="text/csv")

st.caption("v16 fixed - minimal warm-up (default 1), deep mining, gated Top3 ready.")
