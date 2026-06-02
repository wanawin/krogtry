#!/usr/bin/env python3
# BUILD: core025_dynamic_stream_rank_lab_standalone_v2_SCHEMA_AWARE__2026-06-01

from __future__ import annotations
import io, re
import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_dynamic_stream_rank_lab_standalone_v2_SCHEMA_AWARE__2026-06-01"
CORE_MEMBERS = {"0025", "0225", "0255"}

st.set_page_config(page_title="Core025 Dynamic Stream Rank Lab v2", layout="wide")
st.title("Core025 Dynamic Stream Rank Lab — Standalone v2 Schema-Aware")
st.caption(BUILD_MARKER)
st.info("Upload ONLY the all-stream lab CSV. No full history, full truth, promoted library, or daily files required.")

def load_any(uploaded):
    if uploaded is None:
        return pd.DataFrame()
    name = str(getattr(uploaded, "name", "")).lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded)
        raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8", errors="ignore")
        head = raw[:4096].decode("utf-8", errors="ignore")
        sep = "\t" if (name.endswith(".tsv") or (head.count("\t") > head.count(","))) else ","
        return pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
    except Exception as e:
        st.error(f"Could not load file: {e}")
        return pd.DataFrame()

def member_norm(x):
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    for tok in re.findall(r"\d+", str(x)):
        if tok in CORE_MEMBERS:
            return tok
        if tok in {"25", "225", "255"}:
            return {"25": "0025", "225": "0225", "255": "0255"}[tok]
        z = tok.zfill(4)
        if z in CORE_MEMBERS:
            return z
    return ""

def normalize_schema(df):
    out = df.copy()
    aliases = {
        "transition_date": "Date",
        "date": "Date",
        "stream": "StreamKey",
        "stream_key": "StreamKey",
        "winning_member": "TrueMember",
        "true_member": "TrueMember",
        "Top1": "PredictedMember",
        "top1": "PredictedMember",
        "Top2": "Top2_pred",
        "top2": "Top2_pred",
        "Top3": "ThirdMember",
        "top3": "ThirdMember",
        "Top1_score": "score_top1",
        "Top2_score": "score_top2",
        "Top3_score": "score_top3",
    }
    for src, dst in aliases.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    return out

def num(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)

def bool01(df, col):
    if col not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    s = df[col]
    if s.dtype == bool:
        return s.astype(int)
    return s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y", "play"]).astype(int)

def scale_high(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    return ((s - lo) / (hi - lo)).fillna(0.0)

def scale_low(s):
    return 1.0 - scale_high(s)

def validate_input(df):
    missing = []
    for c in ["Date", "StreamKey"]:
        if c not in df.columns:
            missing.append(c)
    if "TrueMember" not in df.columns and "Result" not in df.columns:
        missing.append("TrueMember/winning_member or Result")
    return len(missing) == 0, missing

def infer_true_member(df):
    if "TrueMember" in df.columns:
        s = df["TrueMember"].map(member_norm)
        if s.astype(str).str.len().gt(0).any():
            return s
    if "Result" in df.columns:
        return df["Result"].map(member_norm)
    return pd.Series("", index=df.index)

def add_dynamic_rank(df, weights):
    out = normalize_schema(df)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out[out["Date"].notna()].copy()

    # Score components from both old and new schemas.
    comp = pd.DataFrame(index=out.index)
    comp["top1_score"] = scale_high(num(out, "score_top1", default=np.nan))
    comp["top2_score"] = scale_high(num(out, "score_top2", default=np.nan))
    comp["top3_score"] = scale_high(num(out, "score_top3", default=np.nan))
    comp["gap"] = scale_high(num(out, "gap", default=np.nan))
    comp["ratio_inverse"] = scale_low(num(out, "ratio", default=np.nan))
    comp["compression_inverse"] = scale_low(num(out, "compression_factor", default=np.nan))
    comp["exclusivity"] = scale_high(num(out, "exclusivity_strength", default=np.nan))
    comp["rule_gap_top12"] = scale_high(num(out, "rule_gap_top12", default=np.nan))
    comp["boost_gap_top12"] = scale_high(num(out, "boost_gap_top12", default=np.nan))
    comp["rule_margin_top1_top2"] = scale_high(num(out, "rule_margin_top1_top2", default=np.nan))
    comp["boost_margin_top1_top2"] = scale_high(num(out, "boost_margin_top1_top2", default=np.nan))
    comp["fired_rule_count"] = scale_high(num(out, "fired_rule_count", default=np.nan))
    comp["near_miss_rule_count_inverse"] = scale_low(num(out, "near_miss_rule_count", default=np.nan))
    comp["is_play_top1"] = bool01(out, "is_play_top1")
    comp["is_play_top2"] = bool01(out, "is_play_top2")
    comp["is_skip_inverse"] = 1 - bool01(out, "is_skip")
    comp["play_rule_hit"] = bool01(out, "play_rule_hit")

    # old schema fallbacks, if present
    comp["confidence"] = scale_high(num(out, "ModelConfidenceScore", default=np.nan))
    comp["universal_family"] = scale_high(num(out, "UniversalFamilyScore", default=np.nan))
    comp["row_strength"] = scale_high(num(out, "RowStrengthScore", default=np.nan))
    comp["family_density"] = scale_high(num(out, "FamilyHitDensityScore", default=np.nan))
    comp["hit_density"] = scale_high(num(out, "hit_density", default=np.nan))
    comp["margin"] = scale_high(num(out, "Margin", default=np.nan))

    # Seed traits
    comp["seed_sum"] = scale_high(num(out, "seed_sum", default=np.nan))
    comp["seed_spread"] = scale_high(num(out, "seed_spread", default=np.nan))

    score = pd.Series(0.0, index=out.index)
    for c in comp.columns:
        score += float(weights.get(c, 0.0)) * comp[c]

    out["DynamicDailyStreamScore"] = score
    out["DynamicStreamRank"] = out.groupby("Date")["DynamicDailyStreamScore"].rank(method="first", ascending=False).astype(int)

    out["TrueMember_Normalized"] = infer_true_member(out)
    out["IsCore025Winner"] = out["TrueMember_Normalized"].isin(CORE_MEMBERS).astype(int)

    # Static comparison if present.
    if "rank" in out.columns:
        out["StaticRank_num"] = pd.to_numeric(out["rank"], errors="coerce")
    elif "StreamRank" in out.columns:
        out["StaticRank_num"] = pd.to_numeric(out["StreamRank"], errors="coerce")
    else:
        out["StaticRank_num"] = np.nan
    out["DynamicMinusStaticRank"] = out["DynamicStreamRank"] - out["StaticRank_num"]
    out["MovedVsStaticRank"] = out["DynamicStreamRank"].ne(out["StaticRank_num"]).astype(int)

    return out

def summary_table(ranked):
    winners = ranked[ranked["IsCore025Winner"].eq(1)].copy()
    total = len(winners)
    movement = ranked.groupby("StreamKey").agg(UniqueDynamicRanks=("DynamicStreamRank", "nunique")).reset_index()
    moving = int(movement["UniqueDynamicRanks"].gt(1).sum())
    rows = [
        {"Metric": "Rows processed", "Value": len(ranked), "Pct": "", "Notes": ""},
        {"Metric": "Dates processed", "Value": ranked["Date"].nunique(), "Pct": "", "Notes": ""},
        {"Metric": "Streams processed", "Value": ranked["StreamKey"].nunique(), "Pct": "", "Notes": ""},
        {"Metric": "Avg rows per date", "Value": round(len(ranked)/max(ranked["Date"].nunique(),1), 2), "Pct": "", "Notes": "Should be near all-stream daily count"},
        {"Metric": "Core025 winners", "Value": total, "Pct": "", "Notes": ""},
        {"Metric": "Streams with dynamic rank movement", "Value": moving, "Pct": f"{moving / max(ranked['StreamKey'].nunique(), 1):.2%}", "Notes": ""},
    ]
    for n in [1,3,5,10,15,20,25,30,40,50,60,70,75,78,80,100]:
        cap = int(winners["DynamicStreamRank"].le(n).sum()) if total else 0
        rows.append({"Metric": f"Winners in Dynamic Top {n}", "Value": cap, "Pct": f"{cap/total:.2%}" if total else "", "Notes": ""})
    if ranked["StaticRank_num"].notna().any():
        for n in [10,20,30,40,50,60,70,75,78]:
            cap = int(winners["StaticRank_num"].le(n).sum()) if total else 0
            rows.append({"Metric": f"Winners in Static Rank Top {n}", "Value": cap, "Pct": f"{cap/total:.2%}" if total else "", "Notes": "Comparison"})
    return pd.DataFrame(rows)

def movement_table(ranked):
    return ranked.groupby("StreamKey").agg(
        UniqueDynamicRanks=("DynamicStreamRank", "nunique"),
        MinDynamicRank=("DynamicStreamRank", "min"),
        MaxDynamicRank=("DynamicStreamRank", "max"),
        AvgDynamicRank=("DynamicStreamRank", "mean"),
        Rows=("DynamicStreamRank", "size"),
    ).reset_index().sort_values(["UniqueDynamicRanks","AvgDynamicRank"], ascending=[False, True])

def winner_detail(ranked):
    winners = ranked[ranked["IsCore025Winner"].eq(1)].copy()
    cols = [c for c in [
        "Date","StreamKey","event_id","seed_date","seed","Result","TrueMember","TrueMember_Normalized",
        "DynamicStreamRank","DynamicDailyStreamScore","StaticRank_num","DynamicMinusStaticRank",
        "Top1","Top2","Top3","Top1_score","Top2_score","Top3_score",
        "gap","ratio","dominance_state","play_mode","play_reason",
        "is_play_top1","is_play_top2","is_skip","play_rule_hit",
        "winning_member","top1_hit","top2_hit","top3_hit"
    ] if c in winners.columns]
    return winners[cols].sort_values(["Date","DynamicStreamRank"])

def daily_ranked_view(ranked):
    cols = [c for c in [
        "Date","DynamicStreamRank","DynamicDailyStreamScore","StaticRank_num","DynamicMinusStaticRank",
        "StreamKey","event_id","seed_date","seed",
        "Top1","Top2","Top3","Top1_score","Top2_score","Top3_score",
        "gap","ratio","dominance_state","play_mode","play_reason",
        "is_play_top1","is_play_top2","is_skip","play_rule_hit",
        "TrueMember","winning_member","Result","IsCore025Winner"
    ] if c in ranked.columns]
    return ranked[cols].sort_values(["Date","DynamicStreamRank"])

default_weights = {
    "top1_score": 1.20, "gap": 1.00, "exclusivity": 0.90,
    "rule_gap_top12": 0.85, "boost_gap_top12": 0.75,
    "rule_margin_top1_top2": 0.70, "boost_margin_top1_top2": 0.65,
    "ratio_inverse": 0.55, "compression_inverse": 0.45,
    "fired_rule_count": 0.40, "near_miss_rule_count_inverse": 0.35,
    "is_play_top1": 0.35, "is_play_top2": 0.20, "is_skip_inverse": 0.25,
    "play_rule_hit": 0.20,
    "top2_score": 0.15, "top3_score": 0.05,
    "seed_sum": 0.05, "seed_spread": 0.05,
    # old schema fallback weights
    "confidence": 0.50, "universal_family": 0.30, "row_strength": 0.20,
    "family_density": 0.20, "hit_density": 0.20, "margin": 0.20,
}

with st.sidebar:
    st.header("Input")
    up = st.file_uploader("Upload all-stream lab CSV", type=["csv","txt","tsv","xlsx","xls"])
    st.header("Score weights")
    st.caption("Leave unchanged for baseline.")
    weights = {k: st.number_input(k, value=float(v), step=0.05, format="%.2f") for k, v in default_weights.items()}

df = load_any(up) if up is not None else pd.DataFrame()
if df.empty:
    st.warning("Upload the all-stream lab CSV to begin.")
    st.stop()

df = normalize_schema(df)
ok, missing = validate_input(df)
if not ok:
    st.error("Input is missing required columns after alias mapping: " + ", ".join(missing))
    st.write("Available columns:")
    st.write(list(df.columns))
    st.stop()

st.success(f"Loaded {len(df):,} rows | {pd.to_datetime(df['Date'], errors='coerce').nunique():,} dates | {df['StreamKey'].nunique():,} streams.")
st.caption("Processing waits for the Run button.")

if st.button("Run Dynamic Stream Rank Lab v2", type="primary"):
    ranked = add_dynamic_rank(df, weights)
    st.session_state["summary"] = summary_table(ranked)
    st.session_state["movement"] = movement_table(ranked)
    st.session_state["winners"] = winner_detail(ranked)
    st.session_state["daily"] = daily_ranked_view(ranked)
    st.success("Dynamic Stream Rank Lab v2 complete.")

for title, key, filename, height in [
    ("Dynamic Rank Summary", "summary", "core025_dynamic_rank_v2_summary.csv", 420),
    ("Winner Dynamic Rank Detail", "winners", "core025_dynamic_rank_v2_winner_detail.csv", 900),
    ("Stream Movement Audit", "movement", "core025_dynamic_rank_v2_stream_movement.csv", 600),
    ("Full Daily Dynamic Ranked Table", "daily", "core025_dynamic_rank_v2_full_daily_ranked.csv", 900),
]:
    obj = st.session_state.get(key)
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        st.subheader(title)
        st.dataframe(obj, use_container_width=True, hide_index=True, height=height)
        st.download_button(f"Download {title}", obj.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", use_container_width=True, key=f"download_{key}")
