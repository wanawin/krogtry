#!/usr/bin/env python3
# BUILD: core025_northern_lights__2026-04-23_v32_production_bells_whistles

from __future__ import annotations

import io
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_northern_lights__2026-04-23_v32_production_bells_whistles"
MEMBERS = ["0025", "0225", "0255"]

st.set_page_config(page_title="Core025 Northern Lights", layout="wide")
st.title("Core025 Northern Lights")
st.caption(BUILD_MARKER)
st.warning(
    "Northern Lights Production keeps the exact v22 goal-mode runtime path and adds the bells and whistles: "
    "persistent downloads, progress/status, percentile list, and stream reduction diagnostics. Nothing runs until you click Run."
)

if "nl_prod_results" not in st.session_state:
    st.session_state["nl_prod_results"] = None
if "nl_prod_new_rules" not in st.session_state:
    st.session_state["nl_prod_new_rules"] = None
if "nl_prod_metrics" not in st.session_state:
    st.session_state["nl_prod_metrics"] = None
if "nl_prod_streams" not in st.session_state:
    st.session_state["nl_prod_streams"] = None
if "nl_prod_pct" not in st.session_state:
    st.session_state["nl_prod_pct"] = None


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
    mapping = {"25": "0025", "225": "0225", "255": "0255", "0025": "0025", "0225": "0225", "0255": "0255"}
    return mapping.get(s, s.zfill(4) if s.isdigit() else s)


def ensure_truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if "WinningMember" not in out.columns and "TrueMember" not in out.columns:
        raise ValueError("FULL TRUTH file must contain WinningMember or TrueMember.")
    source_col = "WinningMember" if "WinningMember" in out.columns else "TrueMember"
    out["TrueMember"] = out[source_col].apply(normalize_win)
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


def deep_mine_separators(df: pd.DataFrame, min_rate: float = 0.76, min_support: int = 5) -> pd.DataFrame:
    mined: List[Dict] = []
    trait_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in [
            "pair_has_", "adj_ord_has_", "parity_pattern", "highlow_pattern",
            "repeat_shape", "palindrome", "consec", "mirror", "sum_bucket",
            "spread_bucket", "has", "cnt"
        ])
    ]

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
                            mined.append({
                                "pair": "TRUTH_MINED",
                                "trait_stack": f"{col1}={v1} && {col2}={v2}",
                                "winner_member": m,
                                "winner_rate": rate,
                                "support": int((subset["TrueMember"] == m).sum()),
                                "pair_gap": 0.0,
                                "stack_size": 2,
                            })

    if not mined:
        return pd.DataFrame(columns=["pair", "trait_stack", "winner_member", "winner_rate", "support", "pair_gap", "stack_size"])
    return pd.DataFrame(mined).drop_duplicates(subset=["trait_stack", "winner_member"]).reset_index(drop=True)


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


def run_v22_goal_mode(df: pd.DataFrame, lib_df: pd.DataFrame, full_312_mode: bool, max_plays: int, max_top2: int,
                      min_margin: float, prune_pct: int, trait_weight: float, warm_up: int,
                      truth_min_rate: float, truth_min_support: int) -> Tuple[pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    new_rules = deep_mine_separators(df, min_rate=truth_min_rate, min_support=truth_min_support)
    all_rules = pd.concat([lib_df, new_rules], ignore_index=True) if not new_rules.empty else lib_df.copy()

    if "hit_density" in df.columns:
        hit_density_numeric = pd.to_numeric(df["hit_density"], errors="coerce")
        thresh = hit_density_numeric.quantile(prune_pct / 100.0)
        df = df.copy()
        df["_hit_density_num"] = hit_density_numeric
        df["_kept_after_prune"] = (hit_density_numeric >= thresh).astype(int)
        df_p = df[df["_kept_after_prune"] == 1].copy()
    else:
        df = df.copy()
        df["_hit_density_num"] = pd.NA
        df["_kept_after_prune"] = 1
        df_p = df.copy()

    stream_col = "StreamKey" if "StreamKey" in df.columns else ("stream" if "stream" in df.columns else None)
    if stream_col is not None:
        stream_diag = df.groupby(stream_col, dropna=False).agg(
            rows=("TrueMember", "size"),
            kept_after_prune=("_kept_after_prune", "sum"),
            mean_hit_density=("_hit_density_num", "mean"),
        ).reset_index().rename(columns={stream_col: "StreamKey"})
        stream_diag["pruned_out"] = stream_diag["rows"] - stream_diag["kept_after_prune"]
        stream_diag["kept_pct"] = (stream_diag["kept_after_prune"] / stream_diag["rows"] * 100).round(1)
        stream_diag = stream_diag.sort_values(["mean_hit_density", "rows"], ascending=[False, False]).reset_index(drop=True)
    else:
        stream_diag = pd.DataFrame(columns=["StreamKey", "rows", "kept_after_prune", "mean_hit_density", "pruned_out", "kept_pct"])

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
        seed_str = str(row.get("seed", "")).strip()

        if margin < min_margin:
            gate = False
            if (
                ("0" in seed_str and "9" in seed_str)
                or len(set(seed_str)) <= 2
                or any((d * 2) in seed_str for d in "0123456789")
                or "88" in seed_str or "99" in seed_str or "00" in seed_str
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
            "RecommendedPlay": f"**{top}**",
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

    return pd.DataFrame(results), new_rules, gate_count, stream_diag


with st.sidebar:
    st.header("Inputs")
    truth_file = st.file_uploader("Upload FULL TRUTH file (.csv or .txt)", type=["csv", "txt", "tsv"], key="truth")
    lib_file = st.file_uploader("Upload promoted separator library CSV", type=["csv"], key="lib")

tab_goal, tab_help = st.tabs(["Northern Lights Goal Mode", "Notes"])

with tab_goal:
    if not (truth_file and lib_file):
        st.info("Upload both FULL TRUTH and promoted separator library to continue.")
    else:
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

        run_clicked = st.button("🚀 Run Northern Lights — Exact v22 Goal Mode", type="primary", use_container_width=True)

        if run_clicked:
            status = st.status("Running Northern Lights production...", expanded=True)
            prog = st.progress(0, text="Initializing")
            with status:
                st.write("Loading truth and promoted library inputs")
                prog.progress(10, text="Mining truth-derived separators")
                res_df, new_rules, gate_count, stream_diag = run_v22_goal_mode(
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
                prog.progress(70, text="Computing summaries")
                total = len(res_df)
                t1 = int(res_df["Top1_Correct"].sum()) if total else 0
                nt2 = int(res_df["Needed_Top2"].sum()) if total else 0
                waste = int(res_df["Waste_Top2"].sum()) if total else 0
                miss = int(res_df["Miss"].sum()) if total else 0
                capture = (t1 + nt2) / total * 100 if total > 0 else 0.0
                objective = (t1 * 3.0) + (nt2 * 2.0) - (waste * 1.2) - (miss * 2.5)
                play_top2_rows = int((res_df["Needed_Top2"] == 1).sum())
                pct_df = res_df.copy()
                pct_df["Margin"] = pd.to_numeric(pct_df["Margin"], errors="coerce")
                pct_df["margin_percentile"] = pct_df["Margin"].rank(pct=True, method="average").round(4)
                pct_df = pct_df[["rank", "seed", "PredictedMember", "Top2_pred", "TrueMember", "Margin", "margin_percentile", "Needed_Top2", "Miss"]]

                st.session_state["nl_prod_results"] = res_df
                st.session_state["nl_prod_new_rules"] = new_rules
                st.session_state["nl_prod_metrics"] = {
                    "capture": capture, "total": total, "top1": t1, "needed_top2": nt2,
                    "waste_top2": waste, "miss": miss, "objective": objective,
                    "gate_count": gate_count, "new_rules": int(len(new_rules)),
                    "play_top2_rows": play_top2_rows,
                }
                st.session_state["nl_prod_streams"] = stream_diag
                st.session_state["nl_prod_pct"] = pct_df
                prog.progress(100, text="Done")
                status.update(label="Northern Lights production complete", state="complete", expanded=False)

        if st.session_state["nl_prod_results"] is not None:
            res_df = st.session_state["nl_prod_results"]
            new_rules = st.session_state["nl_prod_new_rules"]
            metrics = st.session_state["nl_prod_metrics"]
            stream_diag = st.session_state["nl_prod_streams"]
            pct_df = st.session_state["nl_prod_pct"]

            st.subheader("Northern Lights — Exact v22 Goal Mode Results")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Capture Rate", f"{metrics['capture']:.1f}%", f"({metrics['top1'] + metrics['needed_top2']}/{metrics['total']})")
                st.metric("Top1 Wins", metrics["top1"])
            with r2:
                st.metric("Needed Top2", metrics["needed_top2"])
                st.metric("Waste Top2", metrics["waste_top2"])
            with r3:
                st.metric("Misses", metrics["miss"])
                st.metric("Objective", f"{metrics['objective']:.1f}")
                st.metric("Total Evaluated", metrics["total"])

            d1, d2, d3 = st.columns(3)
            with d1:
                st.metric("GATED_TOP3 Fired", metrics["gate_count"])
            with d2:
                st.metric("New Separators Mined", metrics["new_rules"])
            with d3:
                st.metric("PLAY_TOP2 Rows", metrics["play_top2_rows"])

            st.subheader("Recommended plays")
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            st.subheader("Percentile list")
            st.dataframe(pct_df, use_container_width=True, hide_index=True)

            st.subheader("Stream reduction diagnostics")
            st.dataframe(stream_diag, use_container_width=True, hide_index=True)

            st.subheader("Downloads")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.download_button("Download Results CSV", res_df.to_csv(index=False).encode("utf-8"),
                                   "walkforward_results__core025_northern_lights__v32_production.csv", "text/csv", use_container_width=True)
            with c2:
                st.download_button("Download Results TXT", res_df.to_csv(index=False).encode("utf-8"),
                                   "walkforward_results__core025_northern_lights__v32_production.txt", "text/plain", use_container_width=True)
            with c3:
                st.download_button("Download New Separators CSV", new_rules.to_csv(index=False).encode("utf-8"),
                                   "new_truth_mined_separators__core025_northern_lights__v32.csv", "text/csv", use_container_width=True)
            with c4:
                st.download_button("Download New Separators TXT", new_rules.to_csv(index=False).encode("utf-8"),
                                   "new_truth_mined_separators__core025_northern_lights__v32.txt", "text/plain", use_container_width=True)

            q1, q2 = st.columns(2)
            with q1:
                st.download_button("Download Percentile List CSV", pct_df.to_csv(index=False).encode("utf-8"),
                                   "percentile_list__core025_northern_lights__v32.csv", "text/csv", use_container_width=True)
            with q2:
                st.download_button("Download Stream Reduction CSV", stream_diag.to_csv(index=False).encode("utf-8"),
                                   "stream_reduction__core025_northern_lights__v32.csv", "text/csv", use_container_width=True)

with tab_help:
    st.markdown("""
**What this build is**
- Northern Lights shell
- Exact v22 runtime path inside it
- Persistent downloads and no reset after one download
- Progress/status bar
- Percentile list
- Stream reduction diagnostics

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
""")
