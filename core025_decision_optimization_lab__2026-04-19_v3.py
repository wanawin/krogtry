#!/usr/bin/env python3
# BUILD: core025_northern_lights__2026-04-22_v31_exact_v22_goal_mode

from __future__ import annotations

import io
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_northern_lights__2026-04-22_v31_exact_v22_goal_mode"
MEMBERS = ["0025", "0225", "0255"]

st.set_page_config(page_title="Core025 Northern Lights", layout="wide")
st.title("Core025 Northern Lights")
st.caption(BUILD_MARKER)
st.warning(
    "This build ports the exact v22 goal-mode runtime path into Northern Lights. "
    "Use FULL TRUTH + promoted separator library here. Nothing runs until you click Run."
)

# ============================================================
# Helpers
# ============================================================

def load_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw), dtype=str)

    if name.endswith(".txt") or name.endswith(".tsv"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str)
        except Exception:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", dtype=str)

    raise ValueError(f"Unsupported file type: {uploaded_file.name}")

def normalize_win(x) -> str:
    if pd.isna(x) or str(x).strip() == "":
        return ""
    s = str(x).strip().replace(" ", "")
    mapping = {
        "25": "0025", "225": "0225", "255": "0255",
        "0025": "0025", "0225": "0225", "0255": "0255"
    }
    return mapping.get(s, s.zfill(4) if s.isdigit() else s)

def ensure_truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if "WinningMember" not in out.columns and "TrueMember" not in out.columns:
        raise ValueError("FULL TRUTH file must contain WinningMember or TrueMember.")
    out["TrueMember"] = out.get("WinningMember", out.get("TrueMember", pd.Series([""] * len(out)))).apply(normalize_win)
    return out

def ensure_library_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    required = ["trait_stack", "winner_member", "winner_rate", "support", "stack_size"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Promoted separator library missing required columns: {missing}")
    out["winner_member"] = out["winner_member"].apply(normalize_win)
    return out

# ============================================================
# Exact v22 deep stacked mining
# ============================================================

def deep_mine_separators(df: pd.DataFrame, min_rate: float = 0.76, min_support: int = 5) -> pd.DataFrame:
    mined: List[Dict] = []

    trait_cols = [
        c for c in df.columns
        if any(
            k in c.lower()
            for k in [
                "pair_has_", "adj_ord_has_", "parity_pattern", "highlow_pattern",
                "repeat_shape", "palindrome", "consec", "mirror", "sum_bucket",
                "spread_bucket", "has", "cnt"
            ]
        )
    ]

    # Single traits
    for col in trait_cols:
        try:
            unique_vals = df[col].astype(str).unique()
        except Exception:
            continue
        for val in unique_vals:
            if val in ["", "nan", "None"]:
                continue
            subset = df[df[col].astype(str) == val]
            if len(subset) < min_support:
                continue
            for m in MEMBERS:
                rate = (subset["TrueMember"] == m).sum() / len(subset)
                if rate >= min_rate:
                    mined.append({
                        "pair": "TRUTH_MINED",
                        "trait_stack": f"{col}={val}",
                        "winner_member": m,
                        "winner_rate": rate,
                        "support": int((subset["TrueMember"] == m).sum()),
                        "pair_gap": 0.0,
                        "stack_size": 1,
                    })

    # 2-trait stacks — exact v22 style narrowed loops
    for i, col1 in enumerate(trait_cols[:25]):
        for col2 in trait_cols[i + 1:i + 20]:
            vals1 = list(df[col1].astype(str).unique())[:12]
            vals2 = list(df[col2].astype(str).unique())[:12]
            for v1 in vals1:
                for v2 in vals2:
                    if v1 in ["", "nan"] or v2 in ["", "nan"]:
                        continue
                    subset = df[(df[col1].astype(str) == v1) & (df[col2].astype(str) == v2)]
                    if len(subset) < min_support:
                        continue
                    for m in MEMBERS:
                        rate = (subset["TrueMember"] == m).sum() / len(subset)
                        if rate >= min_rate + 0.05:
                            stack = f"{col1}={v1} && {col2}={v2}"
                            mined.append({
                                "pair": "TRUTH_MINED",
                                "trait_stack": stack,
                                "winner_member": m,
                                "winner_rate": rate,
                                "support": int((subset["TrueMember"] == m).sum()),
                                "pair_gap": 0.0,
                                "stack_size": 2,
                            })

    if not mined:
        return pd.DataFrame(columns=["pair", "trait_stack", "winner_member", "winner_rate", "support", "pair_gap", "stack_size"])

    out = pd.DataFrame(mined)
    return out.drop_duplicates(subset=["trait_stack", "winner_member"]).reset_index(drop=True)

# ============================================================
# Exact v22 rule application
# ============================================================

def apply_rules(row: pd.Series, rules_df: pd.DataFrame, trait_weight: float) -> Tuple[Dict[str, float], List[str]]:
    boosts = {m: 0.0 for m in MEMBERS}
    fired: List[str] = []

    for _, r in rules_df.iterrows():
        if pd.isna(r.get("trait_stack")):
            continue

        stack = [s.strip() for s in str(r["trait_stack"]).split(" && ")]
        matched = True
        for cond in stack:
            if "=" not in cond:
                continue
            col, val = [x.strip() for x in cond.split("=", 1)]
            if col not in row.index or str(row.get(col, "")).strip() != val:
                matched = False
                break

        if matched:
            winner = normalize_win(r.get("winner_member"))
            if winner in boosts:
                boost = float(r.get("winner_rate", 1.0)) * trait_weight
                boosts[winner] += boost
                fired.append(f"{winner}+{boost:.2f}")

    return boosts, fired

# ============================================================
# Exact v22 runner
# ============================================================

def run_v22_goal_mode(
    df: pd.DataFrame,
    lib_df: pd.DataFrame,
    full_312_mode: bool,
    max_plays: int,
    max_top2: int,
    min_margin: float,
    prune_pct: int,
    trait_weight: float,
    warm_up: int,
    truth_min_rate: float,
    truth_min_support: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    new_rules = deep_mine_separators(df, min_rate=truth_min_rate, min_support=truth_min_support)
    all_rules = pd.concat([lib_df, new_rules], ignore_index=True) if not new_rules.empty else lib_df.copy()

    if "hit_density" in df.columns:
        hit_density_numeric = pd.to_numeric(df["hit_density"], errors="coerce")
        thresh = hit_density_numeric.quantile(prune_pct / 100.0)
        df_p = df[hit_density_numeric >= thresh].copy()
    else:
        df_p = df.copy()

    df_test = df_p.iloc[warm_up:].copy()

    results = []
    top2_count = 0
    max_to_use = len(df_test) if full_312_mode else max_plays
    gate_count = 0

    for i in range(len(df_test)):
        if len(results) >= max_to_use:
            break

        row = df_test.iloc[i]
        boosts, fired = apply_rules(row, all_rules, trait_weight=trait_weight)
        sorted_scores = sorted(boosts.items(), key=lambda x: x[1], reverse=True)

        top = sorted_scores[0][0]
        second = sorted_scores[1][0]
        margin = sorted_scores[0][1] - sorted_scores[1][1]
        true_m = str(row.get("TrueMember", "")).strip()

        # Exact v22 strong gated Top3
        seed_str = str(row.get("seed", "")).strip()

        if margin < min_margin:
            gate = False
            if (
                ("0" in seed_str and "9" in seed_str)
                or len(set(seed_str)) <= 2
                or any((d * 2) in seed_str for d in "0123456789")
                or "88" in seed_str
                or "99" in seed_str
                or "00" in seed_str
            ):
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
            "Fired_Rules": " | ".join(fired[:20]),
        })

    res_df = pd.DataFrame(results)
    return res_df, new_rules, gate_count

# ============================================================
# UI
# ============================================================

with st.sidebar:
    st.header("Inputs")
    truth_file = st.file_uploader("Upload FULL TRUTH file (.csv or .txt)", type=["csv", "txt", "tsv"], key="truth")
    lib_file = st.file_uploader("Upload promoted separator library CSV", type=["csv"], key="lib")

tab_goal, tab_help = st.tabs(["V22 Goal Mode", "Notes"])

with tab_goal:
    if not (truth_file and lib_file):
        st.info("Upload both FULL TRUTH and promoted separator library to continue.")
        st.stop()

    try:
        df = ensure_truth_columns(load_table(truth_file))
        lib_df = ensure_library_columns(pd.read_csv(lib_file, dtype=str))
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("Exact v22 controls")
    full_312_mode = st.checkbox("Full 312 Mode (Backtest All)", value=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        max_plays = st.slider("Max Plays per Day", 20, 100, 40)
        max_top2 = st.slider("Max Top2 per Day", 0, 20, 10)
        min_margin = st.slider("Min Margin for Top2", 0.0, 6.0, 1.0, step=0.1)
    with col2:
        prune_pct = st.slider("Prune Low-Density %", 0, 60, 20)
        seed_boost = st.slider("Seed Boost (v22 UI only; not used in scoring)", 0.0, 12.0, 4.0)
    with col3:
        trait_weight = st.slider("Trait Weight", 0.0, 12.0, 5.0)
        warm_up = st.slider("Warm-up Rows", 0, 5, 1)

    c4, c5 = st.columns(2)
    with c4:
        truth_min_rate = st.slider("Truth-mined min rate", 0.50, 0.95, 0.76, step=0.01)
    with c5:
        truth_min_support = st.slider("Truth-mined min support", 1, 25, 5)

    if st.button("🚀 Run Northern Lights — Exact v22 Goal Mode", type="primary", use_container_width=True):
        res_df, new_rules, gate_count = run_v22_goal_mode(
            df=df,
            lib_df=lib_df,
            full_312_mode=full_312_mode,
            max_plays=int(max_plays),
            max_top2=int(max_top2),
            min_margin=float(min_margin),
            prune_pct=int(prune_pct),
            trait_weight=float(trait_weight),
            warm_up=int(warm_up),
            truth_min_rate=float(truth_min_rate),
            truth_min_support=int(truth_min_support),
        )

        total = len(res_df)
        t1 = int(res_df["Top1_Correct"].sum()) if total else 0
        nt2 = int(res_df["Needed_Top2"].sum()) if total else 0
        waste = int(res_df["Waste_Top2"].sum()) if total else 0
        miss = int(res_df["Miss"].sum()) if total else 0
        capture = (t1 + nt2) / total * 100 if total > 0 else 0.0
        objective = (t1 * 3.0) + (nt2 * 2.0) - (waste * 1.2) - (miss * 2.5)

        st.subheader("Northern Lights — Exact v22 Goal Mode Results")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Capture Rate", f"{capture:.1f}%", f"({t1 + nt2}/{total})")
            st.metric("Top1 Wins", t1)
        with r2:
            st.metric("Needed Top2", nt2)
            st.metric("Waste Top2", waste)
        with r3:
            st.metric("Misses", miss)
            st.metric("Objective", f"{objective:.1f}")
            st.metric("Total Evaluated", f"{total} / {len(df)}")

        d1, d2 = st.columns(2)
        with d1:
            st.metric("GATED_TOP3 Fired", int(gate_count))
        with d2:
            st.metric("New Separators Mined", int(len(new_rules)))

        st.subheader("Walk-forward results")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        st.subheader("Downloads")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.download_button(
                "Download Results CSV",
                data=res_df.to_csv(index=False).encode("utf-8"),
                file_name="walkforward_results__core025_northern_lights__v31_exact_v22_goal_mode.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with k2:
            st.download_button(
                "Download Results TXT",
                data=res_df.to_csv(index=False).encode("utf-8"),
                file_name="walkforward_results__core025_northern_lights__v31_exact_v22_goal_mode.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with k3:
            st.download_button(
                "Download New Separators CSV",
                data=new_rules.to_csv(index=False).encode("utf-8"),
                file_name="new_truth_mined_separators__core025_northern_lights__v31.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with k4:
            st.download_button(
                "Download New Separators TXT",
                data=new_rules.to_csv(index=False).encode("utf-8"),
                file_name="new_truth_mined_separators__core025_northern_lights__v31.txt",
                mime="text/plain",
                use_container_width=True,
            )

with tab_help:
    st.markdown(
        """
**What this build is**
- Northern Lights shell
- Exact v22 runtime path inside it

**Required inputs**
1. `prepared_full_truth_with_stream_stats_v6.csv`
2. promoted separator library CSV

**Important**
- This mode scores exactly like the uploaded v22:
  - matched rule boost = `winner_rate * trait_weight`
  - truth-mined separators are created on the fly each run
  - low-density pruning is applied
  - strong gated Top3 path is applied when margin is below threshold
- `Seed Boost` is shown because it existed in v22 UI, but the uploaded v22 code did not actually use it.
"""
    )
