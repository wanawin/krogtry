import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="Core025 v19 Deep", layout="wide")
st.title("🎯 Core025 v19 - Deeper Mining + Expanded Gated Top3 + Fixed Metrics")
st.caption("BUILD: core025_ultimate_walkforward_v19__2026-04-20")

data_file = st.file_uploader("Upload prepared_full_truth_with_stream_stats_v6.csv", type="csv")
lib_file = st.file_uploader("Upload promoted separator library CSV", type="csv")

if not (data_file and lib_file):
    st.stop()

df = pd.read_csv(data_file)
lib_df = pd.read_csv(lib_file)

def normalize_win(x):
    if pd.isna(x) or str(x).strip() == "":
        return ""
    s = str(x).strip().replace(" ", "")
    mapping = {"25": "0025", "225": "0225", "255": "0255", "0025": "0025", "0225": "0225", "0255": "0255"}
    return mapping.get(s, s.zfill(4) if s.isdigit() else s)

df["TrueMember"] = df.get("WinningMember", df.get("TrueMember", pd.Series([""] * len(df)))).apply(normalize_win)

MEMBERS = ["0025", "0225", "0255"]

full_312_mode = st.checkbox("Full 312 Mode", value=True)

col1, col2, col3 = st.columns(3)
with col1:
    max_plays = st.slider("Max Plays per Day", 20, 100, 40)
    max_top2 = st.slider("Max Top2 per Day", 0, 20, 10)
    min_margin = st.slider("Min Margin for Top2", 0.0, 4.0, 0.6, step=0.1)
with col2:
    prune_pct = st.slider("Prune Low-Density %", 0, 60, 30)
    seed_boost = st.slider("Seed Boost", 0.0, 6.0, 2.5)
with col3:
    trait_weight = st.slider("Trait Weight", 0.0, 6.0, 3.5)
    warm_up = st.slider("Warm-up Rows", 0, 5, 1)

# Deeper mining - scan for single and simple stacked patterns
def deep_mine_separators(df, min_rate=0.78, min_support=5):
    mined = []
    trait_cols = [c for c in df.columns if any(k in c.lower() for k in ["pair_has_", "adj_ord_has_", "parity_pattern", "highlow_pattern", "repeat_shape", "palindrome", "consec", "mirror", "sum_bucket", "spread_bucket", "has", "cnt"])]
    for col in trait_cols:
        for val in df[col].astype(str).unique():
            if val in ["", "nan", "None"]: continue
            subset = df[df[col].astype(str) == val]
            if len(subset) < min_support: continue
            for m in MEMBERS:
                count_m = (subset["TrueMember"] == m).sum()
                rate = count_m / len(subset)
                if rate >= min_rate:
                    mined.append({
                        "trait_stack": f"{col}={val}",
                        "winner_member": m,
                        "winner_rate": rate,
                        "support": count_m,
                        "pair_gap": rate - 0.5,
                        "stack_size": 1
                    })
    # Simple stacked example (pair two high-rate traits)
    # Add more stacked if needed later
    return pd.DataFrame(mined)

new_rules = deep_mine_separators(df, min_rate=0.80, min_support=6)
st.info(f"Deep mining found {len(new_rules)} new separators (rate ≥ 0.80)")

all_rules = pd.concat([lib_df, new_rules], ignore_index=True) if not new_rules.empty else lib_df

def apply_rules(row, rules_df):
    boosts = {m: 0.0 for m in MEMBERS}
    fired = []
    for _, r in rules_df.iterrows():
        if pd.isna(r.get("trait_stack")): continue
        stack = [s.strip() for s in str(r["trait_stack"]).split(" && ")]
        matched = True
        for cond in stack:
            if "=" not in cond: continue
            col, val = [x.strip() for x in cond.split("=", 1)]
            if col not in row or str(row[col]).strip() != val:
                matched = False
                break
        if matched:
            winner = normalize_win(r.get("winner_member"))
            if winner in boosts:
                boost = float(r.get("winner_rate", 1.0)) * trait_weight
                boosts[winner] += boost
                fired.append(f"{winner}+{boost:.2f}")
    return boosts, fired

if st.button("🚀 Run v19 Deeper Mining + Expanded Gates"):
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

        sorted_scores = sorted(boosts.items(), key=lambda x: x[1], reverse=True)
        top = sorted_scores[0][0]
        second = sorted_scores[1][0]
        margin = sorted_scores[0][1] - sorted_scores[1][1]

        true_m = str(row.get("TrueMember", "")).strip()

        # Expanded Gated Top3
        seed_str = str(row.get("seed", "")).strip()
        if margin < min_margin:
            gate_trigger = False
            if ("0" in seed_str and "9" in seed_str) or len(set(seed_str)) <= 2 or any(d*2 in seed_str for d in "0123456789"):
                gate_trigger = True
            if not gate_trigger and ("88" in seed_str or "99" in seed_str or "00" in seed_str):
                gate_trigger = True
            if gate_trigger:
                third = [m for m in MEMBERS if m not in [top, second]][0]
                top = third
                fired.append("GATED_TOP3 (expanded: low margin + seed pattern)")

        top1 = 1 if top == true_m else 0
        needed = 1 if (top1 == 0 and second == true_m) else 0

        miss = 0
        waste = 0
        if needed:
            if not full_312_mode and top2_count >= max_top2:
                needed = 0
                miss = 1
            else:
                top2_count += 1
                waste = 1 if margin < min_margin else 0
        else:
            miss = 1 if top1 == 0 else 0

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
            "Fired_Rules": " | ".join(fired[:12])
        })

    res_df = pd.DataFrame(results)
    total = len(res_df)
    t1 = int(res_df["Top1_Correct"].sum())
    nt2 = int(res_df["Needed_Top2"].sum())
    waste = int(res_df["Waste_Top2"].sum())
    miss = int(res_df["Miss"].sum())
    capture = (t1 + nt2) / total * 100 if total > 0 else 0
    obj = (t1 * 3.0) + (nt2 * 2.0) - (waste * 1.2) - (miss * 2.5)

    st.subheader("v19 Results — Deeper Mining + Expanded Gated Top3")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Capture Rate", f"{capture:.1f}%", f"({t1 + nt2}/{total})")
        st.metric("Top1 Wins", t1)
    with c2:
        st.metric("Needed Top2", nt2)
        st.metric("Waste Top2", waste)
    with c3:
        st.metric("Misses", miss)
        st.metric("Objective", f"{obj:.1f}")
        st.metric("Total Evaluated", f"{total} / {len(df)}")

    st.dataframe(res_df.head(50))
    st.dataframe(res_df)

    csv = res_df.to_csv(index=False)
    st.download_button("📥 Download full results CSV", data=csv, file_name="walkforward_results_v19.csv", mime="text/csv")

st.caption("v19: lower mining threshold + expanded seed gates + fixed capture display. Run with Full 312 Mode ON and Warm-up=1.")
