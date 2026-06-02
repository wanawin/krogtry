#!/usr/bin/env python3
# BUILD: core025_dynamic_stream_rank_lab_standalone__2026-06-01

from __future__ import annotations
import io, re
import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_dynamic_stream_rank_lab_standalone__2026-06-01"
CORE_MEMBERS = {"0025", "0225", "0255"}

st.set_page_config(page_title="Core025 Dynamic Stream Rank Lab", layout="wide")
st.title("Core025 Dynamic Stream Rank Lab — Standalone")
st.caption(BUILD_MARKER)
st.info("Upload ONLY the walkforward per-event file. No full history, full truth, promoted library, or daily files required.")

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
        missing.append("TrueMember or Result")
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
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out[out["Date"].notna()].copy()

    if "StreamKey" not in out.columns:
        if "State" in out.columns and "Game" in out.columns:
            out["StreamKey"] = out["State"].astype(str) + " | " + out["Game"].astype(str)
        else:
            out["StreamKey"] = out.index.astype(str)

    comp = pd.DataFrame(index=out.index)
    comp["confidence"] = scale_high(num(out, "ModelConfidenceScore"))
    comp["universal_family"] = scale_high(num(out, "UniversalFamilyScore"))
    comp["row_strength"] = scale_high(num(out, "RowStrengthScore"))
    comp["family_density"] = scale_high(num(out, "FamilyHitDensityScore"))
    comp["hit_density"] = scale_high(num(out, "hit_density"))
    comp["margin"] = scale_high(num(out, "Margin"))
    comp["top2_risk_inverse"] = scale_low(num(out, "Top2RiskScore"))
    comp["miss_risk_inverse"] = scale_low(num(out, "MissOutlierRiskScore"))
    comp["recommended_stream"] = bool01(out, "RecommendedStream")
    comp["recommended_topn"] = bool01(out, "RecommendedByTopN")
    comp["recommended_rowpct"] = bool01(out, "RecommendedByRowPct")
    comp["top3_rescue"] = bool01(out, "Top3Rescue")
    comp["gated_top3_inverse"] = 1 - bool01(out, "GATED_TOP3")
    comp["seed_sum"] = scale_high(num(out, "seed_sum"))
    comp["seed_spread"] = scale_high(num(out, "seed_spread"))
    comp["seed_has9"] = bool01(out, "seed_has9")
    comp["seed_has0"] = bool01(out, "seed_has0")
    comp["static_stream_rank_anchor"] = scale_low(num(out, "StreamRank", default=np.nan))
    comp["static_single_row_anchor"] = scale_low(num(out, "SingleRow", default=np.nan))

    score = pd.Series(0.0, index=out.index)
    for c in comp.columns:
        score += float(weights.get(c, 0.0)) * comp[c]

    out["DynamicDailyStreamScore"] = score
    out["DynamicStreamRank"] = out.groupby("Date")["DynamicDailyStreamScore"].rank(method="first", ascending=False).astype(int)

    for c in ["StreamRank", "SingleRow", "RowPercentile", "rank", "PlaylistRank", "v79_FinalPlayRank"]:
        if c in out.columns:
            out[c + "_num"] = pd.to_numeric(out[c], errors="coerce")

    if "StreamRank_num" in out.columns:
        out["DynamicMinusStaticStreamRank"] = out["DynamicStreamRank"] - out["StreamRank_num"]
        out["MovedVsStaticStreamRank"] = out["DynamicStreamRank"].ne(out["StreamRank_num"]).astype(int)
    else:
        out["DynamicMinusStaticStreamRank"] = np.nan
        out["MovedVsStaticStreamRank"] = 1

    out["TrueMember_Normalized"] = infer_true_member(out)
    out["IsCore025Winner"] = out["TrueMember_Normalized"].isin(CORE_MEMBERS).astype(int)
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
        {"Metric": "Core025 winners", "Value": total, "Pct": "", "Notes": ""},
        {"Metric": "Streams with dynamic rank movement", "Value": moving, "Pct": f"{moving / max(ranked['StreamKey'].nunique(), 1):.2%}", "Notes": "Unique DynamicStreamRank > 1"},
    ]
    for n in [1,3,5,10,15,20,25,30,40,50,60,70,75,78,80,100]:
        cap = int(winners["DynamicStreamRank"].le(n).sum()) if total else 0
        rows.append({"Metric": f"Winners in Dynamic Top {n}", "Value": cap, "Pct": f"{cap/total:.2%}" if total else "", "Notes": ""})
    if "StreamRank_num" in ranked.columns:
        for n in [10,20,30,40,50,60,70,75,78]:
            cap = int(winners["StreamRank_num"].le(n).sum()) if total else 0
            rows.append({"Metric": f"Winners in Static StreamRank Top {n}", "Value": cap, "Pct": f"{cap/total:.2%}" if total else "", "Notes": "Comparison only"})
    return pd.DataFrame(rows)

def movement_table(ranked):
    agg = {"UniqueDynamicRanks": ("DynamicStreamRank", "nunique"), "MinDynamicRank": ("DynamicStreamRank", "min"), "MaxDynamicRank": ("DynamicStreamRank", "max"), "AvgDynamicRank": ("DynamicStreamRank", "mean"), "Rows": ("DynamicStreamRank", "size")}
    if "StreamRank_num" in ranked.columns:
        agg["StaticStreamRank"] = ("StreamRank_num", "first")
    return ranked.groupby("StreamKey").agg(**agg).reset_index().sort_values(["UniqueDynamicRanks", "AvgDynamicRank"], ascending=[False, True])

def winner_detail(ranked):
    winners = ranked[ranked["IsCore025Winner"].eq(1)].copy()
    cols = [c for c in ["Date","StreamKey","Result","TrueMember","TrueMember_Normalized","DynamicStreamRank","DynamicDailyStreamScore","StreamRank","SingleRow","RowPercentile","rank","DynamicMinusStaticStreamRank","PredictedMember","Top2_pred","ThirdMember","ModelConfidenceScore","UniversalFamilyScore","RowStrengthScore","FamilyHitDensityScore","hit_density","Margin","Top2RiskScore","MissOutlierRiskScore","seed","seed_sum","seed_spread","seed_has9","seed_has0"] if c in winners.columns]
    return winners[cols].sort_values(["Date","DynamicStreamRank"])

def daily_ranked_view(ranked):
    cols = [c for c in ["Date","DynamicStreamRank","DynamicDailyStreamScore","StreamKey","StreamRank","SingleRow","RowPercentile","rank","PredictedMember","Top2_pred","ThirdMember","TrueMember","Result","IsCore025Winner","DynamicMinusStaticStreamRank","ModelConfidenceScore","UniversalFamilyScore","RowStrengthScore","FamilyHitDensityScore","hit_density","Margin","Top2RiskScore","MissOutlierRiskScore","seed","seed_sum","seed_spread","seed_has9","seed_has0"] if c in ranked.columns]
    return ranked[cols].sort_values(["Date","DynamicStreamRank"])

default_weights = {
    "confidence": 1.30, "universal_family": 1.10, "row_strength": 0.90,
    "family_density": 0.80, "hit_density": 0.70, "margin": 0.60,
    "top2_risk_inverse": 0.45, "miss_risk_inverse": 0.35,
    "recommended_stream": 0.35, "recommended_topn": 0.25,
    "recommended_rowpct": 0.25, "top3_rescue": 0.20,
    "gated_top3_inverse": 0.10, "seed_sum": 0.10,
    "seed_spread": 0.10, "seed_has9": 0.05, "seed_has0": 0.05,
    "static_stream_rank_anchor": 0.15, "static_single_row_anchor": 0.15,
}

with st.sidebar:
    st.header("Input")
    up = st.file_uploader("Upload walkforward per-event file only", type=["csv","txt","tsv","xlsx","xls"])
    st.header("Score weights")
    st.caption("Starting lab weights. Leave unchanged for baseline.")
    weights = {k: st.number_input(k, value=float(v), step=0.05, format="%.2f") for k, v in default_weights.items()}

df = load_any(up) if up is not None else pd.DataFrame()
if df.empty:
    st.warning("Upload the walkforward per-event file to begin. No other files are needed.")
    st.stop()

ok, missing = validate_input(df)
if not ok:
    st.error("Input is missing required columns: " + ", ".join(missing))
    st.write("Available columns:")
    st.write(list(df.columns))
    st.stop()

st.success(f"Loaded {len(df):,} rows. Processing waits for the Run button.")

if st.button("Run Dynamic Stream Rank Lab", type="primary"):
    ranked = add_dynamic_rank(df, weights)
    st.session_state["summary"] = summary_table(ranked)
    st.session_state["movement"] = movement_table(ranked)
    st.session_state["winners"] = winner_detail(ranked)
    st.session_state["daily"] = daily_ranked_view(ranked)
    st.success("Dynamic Stream Rank Lab complete.")

for title, key, filename, height in [
    ("Dynamic Rank Summary", "summary", "core025_dynamic_rank_summary.csv", 420),
    ("Winner Dynamic Rank Detail", "winners", "core025_winner_dynamic_rank_detail.csv", 900),
    ("Stream Movement Audit", "movement", "core025_stream_movement_audit.csv", 600),
    ("Full Daily Dynamic Ranked Table", "daily", "core025_full_daily_dynamic_ranked_table.csv", 900),
]:
    obj = st.session_state.get(key)
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        st.subheader(title)
        st.dataframe(obj, use_container_width=True, hide_index=True, height=height)
        st.download_button(f"Download {title}", obj.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", use_container_width=True, key=f"download_{key}")
