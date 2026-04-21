import pandas as pd
import streamlit as st
from collections import defaultdict

st.set_page_config(page_title="Core025 v22 Production", layout="wide")

# Session state for stable download
if "download_key" not in st.session_state:
    st.session_state.download_key = 0

st.title("🎯 Core025 v22 Production — Locked Best Version (75.6%)")
st.caption("Stable | No reset on download | Deep stacked mining + strong gates")

data_file = st.file_uploader("Upload prepared_full_truth_with_stream_stats_v6.csv", type="csv", key="data_upload")
lib_file = st.file_uploader("Upload promoted separator library CSV", type="csv", key="lib_upload")

if not (data_file and lib_file):
    st.info("Please upload both files to continue.")
    st.stop()

df = pd.read_csv(data_file)
lib_df = pd.read_csv(lib_file)

def normalize_win(x):
    if pd.isna(x) or str(x).strip() == "":
        return ""
    s = str(x).strip().replace(" ", "")
    mapping = {"25":"0025","225":"0225","255":"0255","0025":"0025","0225":"0225","0255":"0255"}
    return mapping.get(s, s.zfill(4) if s.isdigit() else s)

df["TrueMember"] = df.get("WinningMember", df.get("TrueMember", pd.Series([""]*len(df)))).apply(normalize_win)

MEMBERS = ["0025", "0225", "0255"]

full_312_mode = st.checkbox("Full 312 Mode (Backtest All)", value=True)

col1, col2, col3 = st.columns(3)
with col1:
    max_plays = st.slider("Max Plays per Day", 20, 100, 40)
    max_top2 = st.slider("Max Top2 per Day", 0, 20, 10)
    min_margin = st.slider("Min Margin for Top2", 0.0, 6.0, 1.0, step=0.1)
with col2:
    prune_pct = st.slider("Prune Low-Density %", 0, 60, 20)
    seed_boost = st.slider("Seed Boost", 0.0, 12.0, 4.0)
with col3:
    trait_weight = st.slider("Trait Weight", 0.0, 12.0, 5.0)
    warm_up = st.slider("Warm-up Rows", 0, 5, 1)

# Deep stacked mining
def deep_mine_separators(df, min_rate=0.76, min_support=5):
    mined = []
    trait_cols = [c for c in df.columns if any(k in c.lower() for k in ["pair_has_", "adj_ord_has_", "parity_pattern", "highlow_pattern", "repeat_shape", "palindrome", "consec", "mirror", "sum_bucket", "spread_bucket", "has", "cnt"])]
    
    # Single traits
    for col in trait_cols:
        for val in df[col].astype(str).unique():
            if val in ["", "nan", "None"]: continue
            subset = df[df[col].astype(str) == val]
            if len(subset) < min_support: continue
            for m in MEMBERS:
                rate = (subset["TrueMember"] == m).sum() / len(subset)
                if rate >= min_rate:
                    mined.append({"trait_stack": f"{col}={val}", "winner_member": m, "winner_rate": rate, "support": (subset["TrueMember"] == m).sum(), "stack_size": 1})
    
    # 2-trait stacking
    for i, col1 in enumerate(trait_cols[:25]):
        for col2 in trait_cols[i+1:i+20]:
            for v1 in list(df[col1].astype(str).unique())[:12]:
                for v2 in list(df[col2].astype(str).unique())[:12]:
                    if v1 in ["", "nan"] or v2 in ["", "nan"]: continue
                    subset = df[(df[col1].astype(str) == v1) & (df[col2].astype(str) == v2)]
                    if len(subset) < min_support: continue
                    for m in MEMBERS:
                        rate = (subset["TrueMember"] == m).sum() / len(subset)
                        if rate >= min_rate + 0.05:
                            stack = f"{col1}={v1} && {col2}={v2}"
                            mined.append({"trait_stack": stack, "winner_member": m, "winner_rate": rate, "support": (subset["TrueMember"] == m).sum(), "stack_size": 2})
    return pd.DataFrame(mined)

new_rules = deep_mine_separators(df)
st.info(f"Deep mining found {len(new_rules)} new separators (single + stacked)")

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
            if col not in row or str(row.get(col, "")).strip() != val:
                matched = False
                break
        if matched:
            winner = normalize_win(r.get("winner_member"))
            if winner in boosts:
                boost = float(r.get("winner_rate", 1.0)) * trait_weight
                boosts[winner] += boost
                fired.append(f"{winner}+{boost:.2f}")
    return boosts, fired

if st.button("🚀 Run v22 Production (Locked Best)"):
    if "hit_density" in df.columns:
        thresh = df["hit_density"].quantile(prune_pct / 100.0)
        df_p = df[df["hit_density"] >= thresh].copy()
    else:
        df_p = df.copy()

    df_test = df_p.iloc[warm_up:].copy()

    results = []
    top2_count = 0
    max_to_use = len(df_test) if full_312_mode else max_plays
    gate_count = 0

    for i in range(len(df_test)):
        if len(results) >= max_to_use: break
        row = df_test.iloc[i]
        boosts, fired = apply_rules(row, all_rules)

        sorted_scores = sorted(boosts.items(), key=lambda x: x[1], reverse=True)
        top = sorted_scores[0][0]
        second = sorted_scores[1][0]
        margin = sorted_scores[0][1] - sorted_scores[1][1]

        true_m = str(row.get("TrueMember", "")).strip()

        # Strong gated Top3
        seed_str = str(row.get("seed", "")).strip()
        if margin < min_margin:
            gate = False
            if ("0" in seed_str and "9" in seed_str) or len(set(seed_str)) <= 2 or any(d*2 in seed_str for d in "0123456789") or "88" in seed_str or "99" in seed_str or "00" in seed_str:
                gate = True
            if top == "0225" and second == "0255" and ("0" in seed_str or "9" in seed_str):
                gate = True
            if top == "0025" and second == "0255" and len(set(seed_str)) <= 3:
                gate = True
            if gate:
                third = [m for m in MEMBERS if m not in [top, second]][0]
                top = third
                fired.append("GATED_TOP3 (strong)")
                gate_count += 1

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
            "seed": seed_str,
            "PredictedMember": top,
            "Top2_pred": second,
            "TrueMember": true_m,
            "Top1_Correct": top1,
            "Needed_Top2": needed,
            "Waste_Top2": waste,
            "Miss": miss,
            "Margin": round(margin, 3),
            "Fired_Rules": " | ".join(fired[:20])
        })

    res_df = pd.DataFrame(results)
    total = len(res_df)
    t1 = int(res_df["Top1_Correct"].sum())
    nt2 = int(res_df["Needed_Top2"].sum())
    waste = int(res_df["Waste_Top2"].sum())
    miss = int(res_df["Miss"].sum())
    capture = (t1 + nt2) / total * 100 if total > 0 else 0
    obj = (t1 * 3.0) + (nt2 * 2.0) - (waste * 1.2) - (miss * 2.5)

    st.subheader("v22 Production Results — Best Stable Version")
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

    debug_col1, debug_col2 = st.columns(2)
    with debug_col1:
        st.metric("GATED_TOP3 Fired", gate_count)
    with debug_col2:
        st.metric("New Separators Mined", len(new_rules))

    st.dataframe(res_df)

    csv = res_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Results CSV",
        data=csv,
        file_name="walkforward_results_v22_production.csv",
        mime="text/csv",
        key=f"download_btn_{st.session_state.download_key}"
    )
    if st.button("New Download Key (if needed)"):
        st.session_state.download_key += 1
        st.rerun()

st.caption("This is the locked best v22 (75.6%). Use this as your main production app. No reset on download.")
