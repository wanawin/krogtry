#!/usr/bin/env python3
# BUILD: core025_northern_lights__2026-05-01_v62_LOCKED_EXECUTION_TOP20

from __future__ import annotations

import io
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_northern_lights__2026-05-01_v62_LOCKED_EXECUTION_TOP20"
MEMBERS = ["0025", "0225", "0255"]

st.set_page_config(page_title="Core025 Northern Lights 025", layout="wide")
st.title("Core025 Northern Lights — 025 Live + Lab")
st.caption(BUILD_MARKER)
st.warning(
    "v62 LOCKED EXECUTION: v47 box/member behavior is treated as the baseline. This build preserves the v45/v47-era member picker, row play-type model, Top3 gating, Top2 cost control, row-by-row percentile scoring, and stream-reduction reports. Straight logic is downstream only and cannot change PredictedMember, Top2_pred, ThirdMember, PlayType, or RecommendedPlay."
)
st.caption("v62 locked defaults: v47 box/member layer first; straight layer only ranks permutations from the already-selected member(s).")

for k in [
    "daily_playlist", "lab_results", "lab_summary", "stream_diag",
    "truth_rules", "loaded_meta", "percentile_df", "stream_method_tests", "row_single_perf", "row_play_model", "straight_recommended", "straight_summary", "straight_ranked", "straight_backtest_events", "straight_backtest_summary", "straight_backtest_depth"
, "daily_playlist_full_visible", "daily_playlist_execution_marked", "daily_playlist_playable", "v62_execution_summary"]:
    if k not in st.session_state:
        st.session_state[k] = None


def load_upload(uploaded_file, header: Optional[int] = "infer") -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw), dtype=str)

    if name.endswith(".txt") or name.endswith(".tsv"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str)
            if df.shape[1] >= 4:
                return df
        except Exception:
            pass
        return pd.read_csv(io.BytesIO(raw), sep="\t", header=None, dtype=str)

    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def extract_pick4_digits(x) -> str:
    if x is None:
        return ""
    s = str(x)
    m = re.search(r"(\d)\D+(\d)\D+(\d)\D+(\d)", s)
    if m:
        return "".join(m.groups())
    m2 = re.search(r"(?<!\d)(\d{4})(?!\d)", s)
    if m2:
        return m2.group(1)
    digits = re.findall(r"\d", s)
    if len(digits) >= 4:
        return "".join(digits[:4])
    return ""


def box_key(s: str) -> str:
    s = extract_pick4_digits(s) or str(s)
    return "".join(sorted(re.sub(r"\D", "", s).zfill(4)[-4:]))


def normalize_member(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().replace("'", "")
    s = re.sub(r"\D", "", s)
    if s in {"25", "025", "0025"}:
        return "0025"
    if s in {"225", "0225"}:
        return "0225"
    if s in {"255", "0255"}:
        return "0255"
    return s.zfill(4) if s else ""


def result_to_core025_member(result4: str) -> str:
    b = box_key(result4)
    return b if b in set(MEMBERS) else ""


def canonical_stream(state: str, game: str) -> str:
    return f"{str(state).strip()} | {str(game).strip()}"


def normalize_history(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    lower = {c.lower(): c for c in df.columns}
    if all(k in lower for k in ["date", "state", "game"]):
        date_col = lower["date"]
        state_col = lower["state"]
        game_col = lower["game"]
        result_col = None
        for cand in ["results", "result", "winning numbers", "winning_numbers", "winningnumbers"]:
            if cand in lower:
                result_col = lower[cand]
                break
        if result_col is None and df.shape[1] >= 4:
            result_col = df.columns[3]
        if result_col is None:
            raise ValueError("History file needs a Results/Result column.")
        out = df[[date_col, state_col, game_col, result_col]].copy()
        out.columns = ["Date", "State", "Game", "Results"]
    else:
        if df.shape[1] < 4:
            raise ValueError("History file must have at least 4 columns: Date, State, Game, Results.")
        out = df.iloc[:, :4].copy()
        out.columns = ["Date", "State", "Game", "Results"]

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Result"] = out["Results"].apply(extract_pick4_digits)
    out = out[out["Date"].notna() & out["Result"].str.len().eq(4)].copy()
    out["Box"] = out["Result"].apply(box_key)
    out["StreamKey"] = out.apply(lambda r: canonical_stream(r["State"], r["Game"]), axis=1)
    out["Core025Member"] = out["Result"].apply(result_to_core025_member)
    out = out.sort_values(["StreamKey", "Date"]).reset_index(drop=True)
    return out


def ensure_truth(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "WinningMember" in df.columns:
        df["TrueMember"] = df["WinningMember"].apply(normalize_member)
    elif "TrueMember" in df.columns:
        df["TrueMember"] = df["TrueMember"].apply(normalize_member)
    else:
        raise ValueError("FULL TRUTH file must contain WinningMember or TrueMember.")

    if "seed" not in df.columns:
        for cand in ["PrevSeed", "feat_seed", "Seed"]:
            if cand in df.columns:
                df["seed"] = df[cand].astype(str)
                break
    if "seed" not in df.columns:
        df["seed"] = ""

    return df


def ensure_library(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = ["trait_stack", "winner_member", "winner_rate", "support", "stack_size"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Promoted separator library missing required columns: {missing}")
    df["winner_member"] = df["winner_member"].apply(normalize_member)
    df["winner_rate"] = pd.to_numeric(df["winner_rate"], errors="coerce").fillna(0.0)
    return df


def digits4(seed: str) -> List[int]:
    s = extract_pick4_digits(seed)
    if len(s) != 4:
        s = re.sub(r"\D", "", str(seed)).zfill(4)[-4:]
    return [int(ch) for ch in s] if len(s) == 4 else [0, 0, 0, 0]


def repeat_shape(d: List[int]) -> str:
    counts = sorted(pd.Series(d).value_counts().tolist(), reverse=True)
    if counts == [1, 1, 1, 1]:
        return "all_unique"
    if counts == [2, 1, 1]:
        return "one_pair"
    if counts == [2, 2]:
        return "two_pair"
    if counts == [3, 1]:
        return "triple"
    if counts == [4]:
        return "quad"
    return "other"


def structure_4(d: List[int]) -> str:
    counts = sorted(pd.Series(d).value_counts().tolist(), reverse=True)
    if counts == [4]:
        return "AAAA"
    if counts == [3, 1]:
        return "AAAB"
    if counts == [2, 2]:
        return "AABB"
    if counts == [2, 1, 1]:
        return "AABC"
    return "ABCD"


def sum_bucket(v: int) -> str:
    if v <= 13:
        return "sum_10_13"
    if v <= 17:
        return "sum_14_17"
    if v <= 21:
        return "sum_18_21"
    return "sum_22_plus"


def spread_bucket(v: int) -> str:
    if v <= 2:
        return "spread_0_2"
    if v <= 4:
        return "spread_3_4"
    if v <= 6:
        return "spread_5_6"
    return "spread_7_plus"


def consec_links_count(d: List[int]) -> int:
    u = sorted(set(d))
    return sum(1 for a, b in zip(u, u[1:]) if b - a == 1)


def enrich_seed_features(df: pd.DataFrame, seed_col: str = "seed") -> pd.DataFrame:
    out = df.copy()
    if seed_col not in out.columns:
        out[seed_col] = ""

    ds = out[seed_col].apply(digits4)
    sums = [sum(d) for d in ds]
    spreads = [max(d) - min(d) for d in ds]

    out["seed_sum"] = sums
    out["seed_spread"] = spreads
    out["seed_even_cnt"] = [sum(1 for x in d if x % 2 == 0) for d in ds]
    out["seed_high_cnt"] = [sum(1 for x in d if x >= 5) for d in ds]
    out["seed_low_cnt"] = [sum(1 for x in d if x <= 4) for d in ds]
    out["seed_has9"] = [1 if 9 in d else 0 for d in ds]
    out["seed_has0"] = [1 if 0 in d else 0 for d in ds]
    out["seed_parity_pattern"] = ["".join("E" if x % 2 == 0 else "O" for x in d) for d in ds]
    out["seed_highlow_pattern"] = ["".join("H" if x >= 5 else "L" for x in d) for d in ds]
    out["seed_repeat_shape"] = [repeat_shape(d) for d in ds]
    out["seed"] = out[seed_col].astype(str).apply(lambda x: extract_pick4_digits(x) or re.sub(r"\D", "", str(x)).zfill(4)[-4:])

    # Critical compatibility layer for the promoted separator library.
    # The promoted library uses unprefixed trait names like cnt0, has0, high, low,
    # parity_pattern, highlow_pattern, structure, sum_bucket, spread_bucket.
    # These are generated from the current seed so daily and lab rule matching actually fires.
    for digit in range(10):
        out[f"cnt{digit}"] = [d.count(digit) for d in ds]
        out[f"has{digit}"] = [1 if digit in d else 0 for d in ds]

    out["even"] = out["seed_even_cnt"]
    out["odd"] = [4 - int(v) for v in out["seed_even_cnt"]]
    out["high"] = out["seed_high_cnt"]
    out["low"] = out["seed_low_cnt"]
    out["parity_pattern"] = out["seed_parity_pattern"]
    out["highlow_pattern"] = out["seed_highlow_pattern"]
    out["structure"] = [structure_4(d) for d in ds]
    out["unique"] = [len(set(d)) for d in ds]
    out["max_rep"] = [max(pd.Series(d).value_counts().tolist()) for d in ds]
    out["consec_links"] = [consec_links_count(d) for d in ds]
    out["pair"] = [1 if max(pd.Series(d).value_counts().tolist()) >= 2 else 0 for d in ds]
    out["sum_bucket"] = [sum_bucket(int(v)) for v in sums]
    out["spread_bucket"] = [spread_bucket(int(v)) for v in spreads]

    return out


def deep_mine_separators(df: pd.DataFrame, min_rate: float = 0.76, min_support: int = 5) -> pd.DataFrame:
    mined: List[Dict] = []
    trait_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in [
            "pair_has_", "adj_ord_has_", "parity_pattern", "highlow_pattern",
            "repeat_shape", "palindrome", "consec", "mirror", "sum_bucket",
            "spread_bucket", "has", "cnt", "seed_sum", "seed_spread"
        ])
    ]

    for col in trait_cols:
        vals = df[col].astype(str).fillna("").unique()
        for val in vals:
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
            vals1 = list(df[col1].astype(str).fillna("").unique())[:12]
            vals2 = list(df[col2].astype(str).fillna("").unique())[:12]
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


def _val_equal(a, b) -> bool:
    sa = str(a).strip()
    sb = str(b).strip()
    if sa == sb:
        return True
    try:
        fa = float(sa)
        fb = float(sb)
        return abs(fa - fb) < 1e-9
    except Exception:
        return False


def apply_rules(row: pd.Series, rules_df: pd.DataFrame, trait_weight: float) -> Tuple[Dict[str, float], List[str]]:
    boosts = {m: 0.0 for m in MEMBERS}
    fired: List[str] = []

    for _, r in rules_df.iterrows():
        stack_text = str(r.get("trait_stack", ""))
        if not stack_text or stack_text.lower() == "nan":
            continue

        # Support both current promoted-library format "a=b && c=d"
        # and older pipe-delimited stacked formats.
        stack = [s.strip() for s in re.split(r"\s*&&\s*|\|", stack_text) if s.strip()]
        matched = True

        for cond in stack:
            if "=" not in cond:
                matched = False
                break
            col, val = [x.strip() for x in cond.split("=", 1)]
            if col not in row.index or not _val_equal(row.get(col, ""), val):
                matched = False
                break

        if matched:
            winner = normalize_member(r.get("winner_member"))
            if winner in boosts:
                boost = float(r.get("winner_rate", 1.0)) * trait_weight
                boosts[winner] += boost
                fired.append(f"{winner}+{boost:.2f}:{stack_text}")
    return boosts, fired


def v22_pick(
    row: pd.Series,
    rules_df: pd.DataFrame,
    trait_weight: float,
    min_margin: float,
    enable_margin_swap: bool = False,
    top2_zone_ratio: float = 0.90,
    top2_zone_margin: float = 5.0,
    top2_risk_threshold: int = 3,
) -> Dict:
    """
    v39 decision layer:
    - Preserve v34/v36 scoring and strict score ranking.
    - Top1 is highest score; Top2 is second-highest score.
    - No third-choice hijack.
    - FIRST gate: evaluate deep Top2-risk traits only if row is already in a Top2 zone.
    """
    boosts, fired = apply_rules(row, rules_df, trait_weight)
    sorted_scores = sorted(boosts.items(), key=lambda x: x[1], reverse=True)

    top = sorted_scores[0][0]
    second = sorted_scores[1][0]
    margin = float(sorted_scores[0][1]) - float(sorted_scores[1][1])
    top1_score = float(sorted_scores[0][1])
    top2_score = float(sorted_scores[1][1])
    ratio = (top2_score / top1_score) if top1_score > 0 else 0.0
    seed = str(row.get("seed", ""))

    margin_swap = 0
    if enable_margin_swap and margin < float(min_margin):
        danger_seed = (
            ("0" in seed and "9" in seed)
            or len(set(seed)) <= 2
            or "00" in seed
            or "88" in seed
            or "99" in seed
        )
        if danger_seed:
            top, second = second, top
            margin_swap = 1
            fired.append("MARGIN_SWAP_TOP1_TOP2_ONLY")

    is_top2_zone = (ratio >= float(top2_zone_ratio)) or (margin <= float(top2_zone_margin))

    risk_score = 0
    risk_reasons = []

    if is_top2_zone:
        if ratio >= float(top2_zone_ratio):
            risk_score += 3
            risk_reasons.append("ratio>=zone")

        if margin <= float(top2_zone_margin):
            risk_score += 3
            risk_reasons.append("margin<=zone")

        if (
            str(row.get("sum_bucket", "")) == "sum_22_plus"
            and str(row.get("spread_bucket", "")) == "spread_3_4"
            and int(float(row.get("seed_has9", 0) or 0)) == 0
        ):
            risk_score += 2
            risk_reasons.append("sum22_spread34_no9")

        if (
            top == "0255"
            and str(row.get("structure", "")) == "AABC"
            and int(float(row.get("seed_has9", 0) or 0)) == 1
        ):
            risk_score += 2
            risk_reasons.append("0255_AABC_has9")

        score_pair = f"{top}>{second}"
        if (
            score_pair == "0225>0255"
            and str(row.get("sum_bucket", "")) == "sum_10_13"
            and int(float(row.get("consec_links", 0) or 0)) == 1
        ):
            risk_score += 2
            risk_reasons.append("0225_0255_sum10_consec1")

        if (
            str(row.get("parity_pattern", "")) == "EOOO"
            and int(float(row.get("pair", 0) or 0)) == 1
        ):
            risk_score += 1
            risk_reasons.append("EOOO_pair")

        if (
            str(row.get("highlow_pattern", "")) == "HLLH"
            and int(float(row.get("seed_has0", 0) or 0)) == 0
        ):
            risk_score += 1
            risk_reasons.append("HLLH_no0")

    top2_decision = "TOP2_REQUIRED" if (is_top2_zone and risk_score >= int(top2_risk_threshold)) else "TOP1_SAFE"

    return {
        "PredictedMember": top,
        "Top2_pred": second,
        "Margin": round(margin, 3),
        "Top2ToTop1Ratio": round(ratio, 4),
        "Top2Zone": int(is_top2_zone),
        "Top2RiskScore": int(risk_score),
        "Top2Decision": top2_decision,
        "Top2RiskReasons": "|".join(risk_reasons),
        "DecisionMode": "STRICT_SCORE_RANK" if margin_swap == 0 else "MARGIN_SWAP_TOP1_TOP2_ONLY",
        "MarginSwap": margin_swap,
        "Fired_Rules": " | ".join(fired[:20]),
        "GATED_TOP3": 0,
        "score_0025": boosts["0025"],
        "score_0225": boosts["0225"],
        "score_0255": boosts["0255"],
    }


def build_live_seed_rows(hist: pd.DataFrame, top_streams: Optional[List[str]] = None) -> pd.DataFrame:
    rows = []
    for stream, g in hist.groupby("StreamKey"):
        g = g.sort_values("Date")
        if g.empty:
            continue
        last = g.iloc[-1]
        rows.append({
            "StreamKey": stream,
            "Date": last["Date"],
            "seed": last["Result"],
            "LastResult": last["Result"],
            "State": last["State"],
            "Game": last["Game"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if top_streams is not None:
        out = out[out["StreamKey"].isin(top_streams)].copy()
    return enrich_seed_features(out, "seed")


def build_lab_event_rows(hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stream, g in hist.groupby("StreamKey"):
        g = g.sort_values("Date").reset_index(drop=True)
        for i in range(1, len(g)):
            true_member = result_to_core025_member(g.loc[i, "Result"])
            if true_member:
                rows.append({
                    "StreamKey": stream,
                    "Date": g.loc[i, "Date"],
                    "seed": g.loc[i - 1, "Result"],
                    "TrueMember": true_member,
                    "Result": g.loc[i, "Result"],
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return enrich_seed_features(out, "seed")


def stream_reduction(hist: pd.DataFrame, window_days: int, prune_pct: int) -> Tuple[pd.DataFrame, List[str]]:
    max_date = hist["Date"].max()
    cutoff = max_date - pd.Timedelta(days=int(window_days))
    win = hist[hist["Date"] >= cutoff].copy()
    grp = win.groupby("StreamKey").agg(
        draws=("Result", "size"),
        core025_hits=("Core025Member", lambda s: int((s != "").sum())),
        last_date=("Date", "max"),
    ).reset_index()
    grp["hit_density"] = grp["core025_hits"] / grp["draws"].replace(0, pd.NA)
    grp["hit_density"] = grp["hit_density"].fillna(0.0)
    thresh = grp["hit_density"].quantile(prune_pct / 100.0) if len(grp) else 0.0
    grp["kept"] = (grp["hit_density"] >= thresh).astype(int)
    grp["rank"] = grp["hit_density"].rank(ascending=False, method="first").astype(int)
    grp["StreamRank"] = grp["rank"]
    grp = grp.sort_values(["kept", "hit_density", "core025_hits"], ascending=[False, False, False]).reset_index(drop=True)
    kept_streams = grp[grp["kept"] == 1]["StreamKey"].tolist()
    return grp, kept_streams

def assign_stream_tiers(stream_df: pd.DataFrame) -> pd.DataFrame:
    out = stream_df.copy()
    if out.empty:
        return out
    n = len(out)
    out = out.sort_values(["StreamRank"], ascending=True).reset_index(drop=True)
    out["RowPercentile"] = ((out["StreamRank"] / n) * 100).round(2)
    out["SingleRow"] = out["StreamRank"].astype(int)
    out["RowBucket10"] = (((out["StreamRank"] - 1) // max(1, int(n / 10))) + 1).clip(1, 10)
    out["StreamTier"] = pd.cut(
        out["RowPercentile"],
        bins=[-0.01, 25, 50, 75, 100],
        labels=["A_TOP25", "B_26_50", "C_51_75", "D_76_100"],
    ).astype(str)
    return out




FALLBACK_ROW_PLAY_COUNTS = {
    1:(6,2,3,0),2:(9,3,4,2),3:(4,2,1,1),4:(6,4,2,0),5:(7,4,2,0),
    6:(7,4,3,0),7:(5,1,3,1),8:(4,1,2,0),9:(4,3,1,0),10:(5,3,2,0),
    11:(3,1,2,0),12:(3,2,1,0),13:(1,1,0,0),14:(8,2,5,1),15:(4,1,2,0),
    16:(2,2,0,0),17:(5,2,1,2),18:(3,2,1,0),19:(3,1,2,0),20:(4,1,2,1),
    21:(5,2,2,1),22:(2,0,1,1),23:(3,3,0,0),24:(12,6,3,3),25:(5,4,0,1),
    26:(5,3,1,0),27:(2,0,1,0),28:(4,3,0,1),29:(5,1,3,0),30:(9,2,5,2),
    31:(1,0,1,0),32:(6,3,2,1),33:(4,1,3,0),34:(6,4,2,0),35:(2,0,2,0),
    36:(4,3,1,0),37:(3,1,2,0),38:(2,1,0,1),39:(2,1,0,1),41:(5,4,0,1),
    42:(3,0,3,0),43:(3,1,2,0),44:(4,1,1,2),45:(4,2,1,0),46:(3,1,2,0),
    47:(5,2,3,0),48:(2,2,0,0),49:(2,1,1,0),50:(6,0,4,2),51:(3,1,1,1),
    52:(5,3,1,1),53:(2,2,0,0),54:(1,1,0,0),55:(9,6,2,1),56:(6,2,1,3),
    57:(2,0,1,1),58:(6,4,0,2),59:(1,0,1,0),60:(6,3,2,1),62:(4,3,1,0),
    63:(2,2,0,0),64:(5,3,1,0),65:(6,5,0,0),66:(6,2,3,1),67:(3,2,1,0),
    68:(3,1,2,0),69:(7,5,1,1),70:(2,2,0,0),71:(4,2,1,1),72:(2,0,1,0),
    73:(1,0,1,0),74:(4,2,2,0),75:(2,1,1,0),76:(5,2,3,0),78:(3,2,1,0),
}


def build_row_play_model(single_row_perf: Optional[pd.DataFrame], min_hits: int, top3_threshold: float, top2_threshold: float, top1_threshold: float, low_top2_threshold: float = 0.35) -> pd.DataFrame:
    source_rows = []
    if single_row_perf is not None and not single_row_perf.empty:
        for _, r in single_row_perf.iterrows():
            try:
                source_rows.append((int(r["SingleRow"]), int(r["rows"]), int(r["top1"]), int(r["top2_captured"]), int(r["top3_captured"])))
            except Exception:
                continue
    else:
        source_rows = [(k, *v) for k, v in FALLBACK_ROW_PLAY_COUNTS.items()]

    rows = []
    for row_num, total, top1, top2, top3 in source_rows:
        if total <= 0:
            continue
        top1_rate = top1 / total
        top2_rate = top2 / total
        top3_rate = top3 / total
        if total < int(min_hits):
            # v44 correction: low sample stays conservative, not automatic Top3.
            play_type = "TOP1_TOP2"
            reason = "LOW_SAMPLE_CONSERVATIVE"
        elif total >= int(min_hits) and top3 > 0 and top3_rate >= float(top3_threshold):
            play_type = "TOP1_TOP2_TOP3"
            reason = "ROW_TOP3_RATE"
        elif top2_rate < float(low_top2_threshold):
            play_type = "TOP1_ONLY"
            reason = "LOW_TOP2_RATE_OPTIMIZED"
        elif top2_rate >= float(top2_threshold) and top2 >= top1:
            play_type = "TOP1_TOP2"
            reason = "ROW_TOP2_HEAVY"
        elif top1_rate >= float(top1_threshold) and top2 == 0 and top3 == 0:
            play_type = "TOP1_ONLY"
            reason = "ROW_TOP1_DOMINANT"
        else:
            play_type = "TOP1_TOP2"
            reason = "ROW_DEFAULT_TOP2"

        rows.append({
            "SingleRow": row_num,
            "RowModelHits": total,
            "RowModelTop1": top1,
            "RowModelTop2": top2,
            "RowModelTop3": top3,
            "RowTop1Rate": round(top1_rate, 4),
            "RowTop2Rate": round(top2_rate, 4),
            "RowTop3Rate": round(top3_rate, 4),
            "RowLowTop2Threshold": float(low_top2_threshold),
            "RowPlayType": play_type,
            "RowPlayTypeReason": reason,
        })
    return pd.DataFrame(rows).sort_values("SingleRow").reset_index(drop=True)


def apply_row_playtype_model(df: pd.DataFrame, row_model_df: Optional[pd.DataFrame], enabled: bool) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if not enabled or row_model_df is None or row_model_df.empty or "SingleRow" not in out.columns:
        out["RowPlayType"] = ""
        out["RowPlayTypeReason"] = "row_model_disabled"
        return out

    cols = ["SingleRow","RowModelHits","RowModelTop1","RowModelTop2","RowModelTop3","RowTop1Rate","RowTop2Rate","RowTop3Rate","RowLowTop2Threshold","RowPlayType","RowPlayTypeReason"]
    out = out.merge(row_model_df[cols], on="SingleRow", how="left")
    out["RowPlayType"] = out["RowPlayType"].fillna("TOP1_TOP2")
    out["RowPlayTypeReason"] = out["RowPlayTypeReason"].fillna("row_unseen_default_top2")
    out["PlayType"] = out["RowPlayType"]
    return out


def add_rare_top3_and_universal_scores(
    df: pd.DataFrame,
    enable_top3_rescue: bool,
    top3_rescue_min_score: int,
    w_model: float,
    w_row: float,
    w_density: float,
    w_outlier: float,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    # Third member = the omitted member from the strict Top1/Top2 pair.
    def _third(row):
        top1 = str(row.get("PredictedMember", ""))
        top2 = str(row.get("Top2_pred", ""))
        rest = [m for m in MEMBERS if m not in {top1, top2}]
        return rest[0] if rest else ""

    out["ThirdMember"] = out.apply(_third, axis=1)

    danger_rows = {56, 44, 17, 50, 58, 24, 30, 2}
    danger_parity = {"EOEO", "OOOO", "OEEE", "EOOE"}
    danger_highlow = {"LLLH", "HHLL", "HLLH"}

    out["Top3RescueScore"] = 0
    out["Top3RescueReasons"] = ""

    reasons_col = []
    scores = []
    for _, r in out.iterrows():
        score = 0
        reasons = []

        row_num = int(float(r.get("SingleRow", 0) or 0))
        if row_num in danger_rows:
            score += 2
            reasons.append("danger_row")

        if str(r.get("ThirdMember", "")) == "0255":
            score += 2
            reasons.append("third_0255")

        if str(r.get("parity_pattern", "")) in danger_parity:
            score += 1
            reasons.append("danger_parity")

        if str(r.get("highlow_pattern", "")) in danger_highlow:
            score += 1
            reasons.append("danger_highlow")

        if str(r.get("sum_bucket", "")) == "sum_18_21":
            score += 1
            reasons.append("sum_18_21")

        if str(r.get("StreamTier", "")) in {"B_26_50", "C_51_75"}:
            score += 1
            reasons.append("mid_tier_miss_zone")

        scores.append(score)
        reasons_col.append("|".join(reasons))

    out["Top3RescueScore"] = scores
    out["Top3RescueReasons"] = reasons_col
    out["Top3Rescue"] = ((out["Top3RescueScore"] >= int(top3_rescue_min_score)) & bool(enable_top3_rescue)).astype(int)

    # Universal score: normalized to 0-100 and family-comparable.
    # Family apps can use the same column names and weights.
    score_cols = [c for c in ["score_0025", "score_0225", "score_0255"] if c in out.columns]
    if score_cols:
        out["ModelConfidenceRaw"] = out[score_cols].max(axis=1)
        max_model = pd.to_numeric(out["ModelConfidenceRaw"], errors="coerce").max()
        out["ModelConfidenceScore"] = (pd.to_numeric(out["ModelConfidenceRaw"], errors="coerce") / max_model * 100).fillna(0) if max_model and max_model > 0 else 0
    else:
        out["ModelConfidenceScore"] = 0

    out["RowStrengthScore"] = (100 - pd.to_numeric(out.get("RowPercentile", 100), errors="coerce")).clip(0, 100).fillna(0)

    dens = pd.to_numeric(out.get("hit_density", 0), errors="coerce").fillna(0)
    max_dens = dens.max()
    out["FamilyHitDensityScore"] = (dens / max_dens * 100) if max_dens and max_dens > 0 else 0

    out["MissOutlierRiskScore"] = (out["Top3RescueScore"] * 10).clip(0, 100)

    total_weight = float(w_model) + float(w_row) + float(w_density) + float(w_outlier)
    if total_weight <= 0:
        total_weight = 1.0

    out["UniversalFamilyScore"] = (
        (out["ModelConfidenceScore"] * float(w_model))
        + (out["RowStrengthScore"] * float(w_row))
        + (out["FamilyHitDensityScore"] * float(w_density))
        - (out["MissOutlierRiskScore"] * float(w_outlier))
    ) / total_weight
    out["UniversalFamilyScore"] = out["UniversalFamilyScore"].round(2)

    return out


def apply_play_recommendation(df: pd.DataFrame, play_policy: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    if "PlayType" not in out.columns or out["PlayType"].isna().all() or (out["PlayType"].astype(str).str.len() == 0).all():
        if play_policy.startswith("Full capture"):
            out["PlayType"] = "TOP1_TOP2"
        elif play_policy.startswith("Top2 risk"):
            out["PlayType"] = out["Top2Decision"].apply(lambda x: "TOP1_TOP2" if str(x) == "TOP2_REQUIRED" else "TOP1_ONLY")
        else:
            out["PlayType"] = "TOP1_ONLY"
        if "Top3Rescue" in out.columns:
            out.loc[out["Top3Rescue"] == 1, "PlayType"] = "TOP1_TOP2_TOP3"

    def _bold_play(row):
        top1 = str(row.get("PredictedMember", ""))
        top2 = str(row.get("Top2_pred", ""))
        top3 = str(row.get("ThirdMember", ""))
        if row["PlayType"] == "TOP1_TOP2_TOP3":
            parts = [f"**{top1}**", f"**{top2}**", f"**{top3}**"]
        elif row["PlayType"] == "TOP1_TOP2":
            parts = [f"**{top1}**", f"**{top2}**"]
        else:
            parts = [f"**{top1}**"]
        return " + ".join([p for p in parts if p and p != "****"])

    out["RecommendedPlay"] = out.apply(_bold_play, axis=1)
    out["Top1_Display"] = out["PredictedMember"].apply(lambda x: f"**{x}**")
    out["Top2_Display"] = out.apply(lambda r: f"**{r['Top2_pred']}**" if r["PlayType"] in ["TOP1_TOP2", "TOP1_TOP2_TOP3"] else str(r["Top2_pred"]), axis=1)
    out["Top3_Display"] = out.apply(lambda r: f"**{r['ThirdMember']}**" if r["PlayType"] == "TOP1_TOP2_TOP3" else str(r.get("ThirdMember", "")), axis=1)
    return out


def add_stream_selection_flags(df: pd.DataFrame, recommended_top_n: int, row_pct_cutoff: int) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["RecommendedByTopN"] = (pd.to_numeric(out["StreamRank"], errors="coerce") <= int(recommended_top_n)).astype(int)
    out["RecommendedByRowPct"] = (pd.to_numeric(out["RowPercentile"], errors="coerce") <= int(row_pct_cutoff)).astype(int)
    out["RecommendedStream"] = ((out["RecommendedByTopN"] == 1) | (out["RecommendedByRowPct"] == 1)).astype(int)
    return out


def summarize_policy(df: pd.DataFrame, mask, label: str, play_policy: str) -> Dict:
    sub = df[mask].copy()
    total = len(sub)
    if total == 0:
        return {
            "method": label, "rows": 0, "top1": 0, "top2_captured": 0, "top3_captured": 0,
            "captured": 0, "miss": 0, "capture_pct": 0.0, "plays": 0, "plays_per_capture": None
        }

    top1 = int(sub["Top1_Correct"].sum())
    top2_all = int(sub["Needed_Top2"].sum())

    if "PlayType" in sub.columns:
        top2_captured = int(((sub["Needed_Top2"] == 1) & (sub["PlayType"].isin(["TOP1_TOP2", "TOP1_TOP2_TOP3"]))).sum())
        top3_captured = int(((sub["Miss"] == 1) & (sub["PlayType"] == "TOP1_TOP2_TOP3")).sum())
        plays = int((sub["PlayType"] == "TOP1_ONLY").sum() + 2*(sub["PlayType"] == "TOP1_TOP2").sum() + 3*(sub["PlayType"] == "TOP1_TOP2_TOP3").sum())
    elif play_policy.startswith("Full capture"):
        top2_captured = top2_all
        top3_captured = 0
        plays = total * 2
    elif play_policy.startswith("Top2 risk"):
        top2_captured = int(((sub["Needed_Top2"] == 1) & (sub["Top2Decision"] == "TOP2_REQUIRED")).sum())
        top3_captured = 0
        plays = total + int((sub["Top2Decision"] == "TOP2_REQUIRED").sum())
    else:
        top2_captured = 0
        top3_captured = 0
        plays = total

    captured = top1 + top2_captured + top3_captured
    miss = total - captured
    return {
        "method": label,
        "rows": total,
        "top1": top1,
        "top2_captured": top2_captured,
        "top3_captured": top3_captured,
        "captured": captured,
        "miss": miss,
        "capture_pct": round((captured / total * 100), 2) if total else 0.0,
        "plays": int(plays),
        "plays_per_capture": round((plays / captured), 3) if captured else None,
    }


def build_stream_method_tests(res: pd.DataFrame, play_policy: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if res.empty or "StreamRank" not in res.columns:
        return pd.DataFrame(), pd.DataFrame()

    work = res.copy()
    work["StreamRank"] = pd.to_numeric(work["StreamRank"], errors="coerce")
    max_rank = int(work["StreamRank"].max()) if work["StreamRank"].notna().any() else 0
    rows = []

    rows.append(summarize_policy(work, work["StreamRank"].notna(), "ALL_STREAMS", play_policy))

    for n in [10, 15, 20, 25, 30, 40, 50, 60]:
        if n <= max_rank:
            rows.append(summarize_policy(work, work["StreamRank"] <= n, f"TOP_{n}_STREAM_ROWS", play_policy))

    for pct in [10, 15, 20, 25, 30, 40, 50, 60, 70, 80]:
        cutoff = max(1, int(round(max_rank * pct / 100)))
        rows.append(summarize_policy(work, work["StreamRank"] <= cutoff, f"ROW_PERCENTILE_TOP_{pct}", play_policy))

    if "StreamTier" in work.columns:
        for tier in ["A_TOP25", "B_26_50", "C_51_75", "D_76_100"]:
            rows.append(summarize_policy(work, work["StreamTier"] == tier, f"STREAM_TIER_{tier}", play_policy))

    method_df = pd.DataFrame(rows)

    single_rows = []
    for rnk, g in work.groupby("StreamRank", dropna=True):
        single_rows.append(summarize_policy(g, pd.Series([True] * len(g), index=g.index), f"SINGLE_ROW_{int(rnk)}", play_policy))
    single_df = pd.DataFrame(single_rows)
    if not single_df.empty:
        single_df["SingleRow"] = single_df["method"].str.replace("SINGLE_ROW_", "", regex=False).astype(int)
        single_df = single_df.sort_values("SingleRow")
    return method_df, single_df



# ================================
# v62 DOWNSTREAM-ONLY STRAIGHT LAYER
# This layer must never alter the locked v47/v45 box/member picker outputs.
# ================================

def v62_unique_permutations(member: str) -> List[str]:
    s = normalize_member(member)
    if s not in MEMBERS:
        return []
    out = sorted({"".join(p) for p in __import__("itertools").permutations(s, 4)})
    return out

def v62_member_order_from_playlist_row(row: pd.Series) -> List[str]:
    ordered = []
    for col in ["PredictedMember", "Top2_pred", "ThirdMember"]:
        val = normalize_member(row.get(col, ""))
        if val in MEMBERS and val not in ordered:
            ordered.append(val)

    play_type = str(row.get("PlayType", row.get("RowPlayType", ""))).upper()
    if "TOP1_ONLY" in play_type:
        return ordered[:1]
    if "TOP1_TOP2_TOP3" in play_type:
        return ordered[:3]
    if "TOP1_TOP2" in play_type:
        return ordered[:2]
    return ordered[:1]

def v62_build_order_reference_tables(hist: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    h = hist.copy()
    h["Result"] = h["Result"].astype(str).str.zfill(4)
    core = h[h["Core025Member"].isin(MEMBERS)].copy()
    if core.empty:
        return {
            "position": pd.DataFrame(),
            "ordered_pair": pd.DataFrame(),
            "stream_order": pd.DataFrame(),
            "recent_order": pd.DataFrame(),
            "repeat_pos": pd.DataFrame(),
        }

    pos_rows = []
    for _, r in core.iterrows():
        member = r["Core025Member"]
        res = r["Result"]
        for i, d in enumerate(res, start=1):
            pos_rows.append({"Member": member, "Pos": i, "Digit": d})
    pos = pd.DataFrame(pos_rows)
    if not pos.empty:
        pos = pos.value_counts(["Member", "Pos", "Digit"]).reset_index(name="Count")
        pos["Total"] = pos.groupby(["Member", "Pos"])["Count"].transform("sum")
        pos["Rate"] = pos["Count"] / pos["Total"]

    pair_rows = []
    for _, r in core.iterrows():
        member = r["Core025Member"]
        res = r["Result"]
        for i in range(3):
            pair_rows.append({"Member": member, "PairPos": i + 1, "Pair": res[i:i+2]})
    pair = pd.DataFrame(pair_rows)
    if not pair.empty:
        pair = pair.value_counts(["Member", "PairPos", "Pair"]).reset_index(name="Count")
        pair["Total"] = pair.groupby(["Member", "PairPos"])["Count"].transform("sum")
        pair["Rate"] = pair["Count"] / pair["Total"]

    stream_order = core.value_counts(["StreamKey", "Core025Member", "Result"]).reset_index(name="Count")
    if not stream_order.empty:
        stream_order["Total"] = stream_order.groupby(["StreamKey", "Core025Member"])["Count"].transform("sum")
        stream_order["Rate"] = stream_order["Count"] / stream_order["Total"]

    recent = core.sort_values("Date").tail(60).copy()
    recent_order = recent.value_counts(["Core025Member", "Result"]).reset_index(name="Count")
    if not recent_order.empty:
        recent_order["Total"] = recent_order.groupby(["Core025Member"])["Count"].transform("sum")
        recent_order["Rate"] = recent_order["Count"] / recent_order["Total"]

    rep_rows = []
    for _, r in core.iterrows():
        res = r["Result"]
        member = r["Core025Member"]
        counts = {d: res.count(d) for d in set(res)}
        double_digits = [d for d, c in counts.items() if c == 2]
        if double_digits:
            d = double_digits[0]
            pos_key = "-".join(str(i+1) for i, x in enumerate(res) if x == d)
            rep_rows.append({"Member": member, "DupPattern": pos_key})
    repeat_pos = pd.DataFrame(rep_rows)
    if not repeat_pos.empty:
        repeat_pos = repeat_pos.value_counts(["Member", "DupPattern"]).reset_index(name="Count")
        repeat_pos["Total"] = repeat_pos.groupby("Member")["Count"].transform("sum")
        repeat_pos["Rate"] = repeat_pos["Count"] / repeat_pos["Total"]

    return {
        "position": pos,
        "ordered_pair": pair,
        "stream_order": stream_order,
        "recent_order": recent_order,
        "repeat_pos": repeat_pos,
    }

def v62_lookup_rate(df: pd.DataFrame, filters: Dict[str, str], default: float = 0.0) -> float:
    if df is None or df.empty:
        return default
    sub = df.copy()
    for k, v in filters.items():
        if k not in sub.columns:
            return default
        sub = sub[sub[k].astype(str).eq(str(v))]
        if sub.empty:
            return default
    return float(sub.iloc[0].get("Rate", default) or default)

def v62_dup_pattern(perm: str) -> str:
    s = str(perm).zfill(4)
    for d in sorted(set(s)):
        if s.count(d) == 2:
            return "-".join(str(i+1) for i, x in enumerate(s) if x == d)
    return ""

def v62_score_perm(perm: str, member: str, row: pd.Series, refs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    perm = str(perm).zfill(4)
    member = normalize_member(member)
    stream = str(row.get("StreamKey", ""))

    # Member confidence is LOCKED to the existing member order only.
    if member == normalize_member(row.get("PredictedMember", "")):
        member_score = 100.0
        member_source = "TOP1"
    elif member == normalize_member(row.get("Top2_pred", "")):
        member_score = 55.0
        member_source = "TOP2"
    else:
        member_score = 25.0
        member_source = "TOP3"

    pos_rates = []
    for i, d in enumerate(perm, start=1):
        pos_rates.append(v62_lookup_rate(refs["position"], {"Member": member, "Pos": i, "Digit": d}))
    position_score = 100.0 * (sum(pos_rates) / len(pos_rates)) if pos_rates else 0.0

    pair_rates = []
    for i in range(3):
        pair_rates.append(v62_lookup_rate(refs["ordered_pair"], {"Member": member, "PairPos": i + 1, "Pair": perm[i:i+2]}))
    ordered_pair_score = 100.0 * (sum(pair_rates) / len(pair_rates)) if pair_rates else 0.0

    stream_order_score = 100.0 * v62_lookup_rate(refs["stream_order"], {"StreamKey": stream, "Core025Member": member, "Result": perm})
    recent_order_score = 100.0 * v62_lookup_rate(refs["recent_order"], {"Core025Member": member, "Result": perm})

    dup = v62_dup_pattern(perm)
    repeat_placement_score = 100.0 * v62_lookup_rate(refs["repeat_pos"], {"Member": member, "DupPattern": dup})

    # v47-style weighting from uploaded straight reports:
    # 25% member + 25% position + 20% ordered pair + 10% stream + 5% recent + 15% repeat/structure proxy.
    confidence = (
        0.25 * member_score
        + 0.25 * position_score
        + 0.20 * ordered_pair_score
        + 0.10 * stream_order_score
        + 0.05 * recent_order_score
        + 0.15 * repeat_placement_score
    )

    evidence = int((len(refs["position"]) if refs["position"] is not None else 0) + (len(refs["ordered_pair"]) if refs["ordered_pair"] is not None else 0))

    return {
        "StraightMemberSource": member_source,
        "MemberConfidenceScore": round(member_score, 2),
        "PositionScore": round(position_score, 2),
        "OrderedPairScore": round(ordered_pair_score, 2),
        "StreamOrderScore": round(stream_order_score, 2),
        "RecentOrderScore": round(recent_order_score, 2),
        "RepeatPlacementScore": round(repeat_placement_score, 2),
        "StraightConfidenceScore": round(confidence, 2),
        "EvidenceSupport": evidence,
        "DupPattern": dup,
    }

def v62_generate_downstream_straights(playlist: pd.DataFrame, hist: pd.DataFrame, recommended_only: bool = True, straight_depth: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if playlist is None or playlist.empty:
        return pd.DataFrame(), pd.DataFrame()

    base = playlist.copy()
    if recommended_only and "RecommendedStream" in base.columns:
        base = base[pd.to_numeric(base["RecommendedStream"], errors="coerce").fillna(0).astype(int).eq(1)].copy()

    refs = v62_build_order_reference_tables(hist)
    rows = []
    summary = []

    for _, row in base.iterrows():
        members = v62_member_order_from_playlist_row(row)
        candidate_rows = []
        for member in members:
            for perm in v62_unique_permutations(member):
                score = v62_score_perm(perm, member, row, refs)
                candidate_rows.append({
                    "PlaylistRank": row.get("PlaylistRank", ""),
                    "RecommendedStream": row.get("RecommendedStream", ""),
                    "StreamRank": row.get("StreamRank", ""),
                    "RowPercentile": row.get("RowPercentile", ""),
                    "SingleRow": row.get("SingleRow", ""),
                    "StreamTier": row.get("StreamTier", ""),
                    "StreamKey": row.get("StreamKey", ""),
                    "State": row.get("State", ""),
                    "Game": row.get("Game", ""),
                    "seed": row.get("LastResult", ""),
                    "BoxPlayType": row.get("PlayType", ""),
                    "BoxRecommendedPlay": row.get("RecommendedPlay", ""),
                    "PredictedMember": row.get("PredictedMember", ""),
                    "Top2_pred": row.get("Top2_pred", ""),
                    "ThirdMember": row.get("ThirdMember", ""),
                    "StraightMember": member,
                    "StraightPermutation": perm,
                    **score,
                    "StraightScoreFormula": "v62 downstream-only: 25% locked member + 25% position + 20% ordered_pair + 10% stream_order + 5% recent_order + 15% repeat_placement",
                })

        cand = pd.DataFrame(candidate_rows)
        if cand.empty:
            continue
        cand = cand.sort_values(["StraightConfidenceScore", "MemberConfidenceScore", "PositionScore"], ascending=[False, False, False]).reset_index(drop=True)
        cand["StreamStraightRank"] = cand.index + 1
        cand["RecommendedStraight"] = (cand["StreamStraightRank"] <= int(straight_depth)).astype(int)
        cand["RecommendedStraightDisplay"] = cand["StraightPermutation"].where(cand["RecommendedStraight"].eq(0), "**" + cand["StraightPermutation"] + "**")
        rows.append(cand)

        rec = cand[cand["RecommendedStraight"].eq(1)].copy()
        summary.append({
            "StreamKey": row.get("StreamKey", ""),
            "eligible_members": ",".join(members),
            "total_candidate_straights": len(cand),
            "recommended_straights": len(rec),
            "best_straight": cand.iloc[0]["StraightPermutation"],
            "best_confidence": cand.iloc[0]["StraightConfidenceScore"],
            "box_play_type": row.get("PlayType", ""),
            "box_play": row.get("RecommendedPlay", ""),
            "stream_rank": row.get("StreamRank", ""),
            "row_percentile": row.get("RowPercentile", ""),
        })

    ranked = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not ranked.empty:
        ranked = ranked.sort_values(["PlaylistRank", "StreamStraightRank"]).reset_index(drop=True)
        ranked["OverallStraightRank"] = ranked.index + 1

    return ranked, pd.DataFrame(summary)


with st.sidebar:
    st.header("Inputs")
    hist_file = st.file_uploader("Upload FULL HISTORY (.txt/.csv)", type=["txt", "csv", "tsv"], key="hist")
    last24_file = st.file_uploader("Upload LAST 24 HOURS optional (.txt/.csv)", type=["txt", "csv", "tsv"], key="last24")
    truth_file = st.file_uploader("Upload FULL TRUTH intelligence file (.csv/.txt)", type=["csv", "txt", "tsv"], key="truth")
    lib_file = st.file_uploader("Upload promoted separator library CSV", type=["csv"], key="lib")

    st.header("Defaults")
    window_days = st.slider("Stream reduction window days", 30, 730, 180)
    prune_pct = st.slider("Prune low-density %", 0, 60, 20)
    trait_weight = st.slider("Trait Weight", 0.0, 12.0, 5.0)
    min_margin = st.slider("Min Margin for cautious swap", 0.0, 6.0, 1.0, step=0.1)
    enable_margin_swap = st.checkbox("Allow cautious Top1/Top2 swap in close-margin danger seeds", value=False)

    st.header("Top2 Risk Gate")
    top2_zone_ratio = st.slider("Top2 zone ratio trigger", 0.50, 1.00, 0.90, step=0.01)
    top2_zone_margin = st.slider("Top2 zone margin trigger", 0.0, 50.0, 5.0, step=0.5)
    top2_risk_threshold = st.slider("Require Top2 if risk score >=", 0, 12, 3, step=1)

    st.header("Operational Display")
    play_policy = st.selectbox(
        "Play recommendation policy",
        [
            "Full capture: Top1+Top2 shown as recommended",
            "Top2 risk gate: Top2 only when TOP2_REQUIRED",
            "Top1 only: cheapest baseline",
        ],
        index=0,
    )
    recommended_top_n = st.slider("Recommended stream Top-N marker", 5, 78, 50, step=1)
    row_pct_cutoff = st.slider("Recommended row percentile marker", 5, 100, 60, step=5)

    st.header("Rare Top3 Rescue")
    enable_top3_rescue = st.checkbox("Enable rare Top3 rescue for miss/outlier rows", value=True)
    top3_rescue_min_score = st.slider("Top3 rescue minimum score", 1, 10, 3, step=1)

    st.header("Row-Based Play Type Model")
    enable_row_playtype_model = st.checkbox("Enable row-based play-type model", value=True)
    row_model_min_hits = st.slider("Row model minimum historical hits", 1, 12, 4, step=1)
    row_top3_rate_threshold = st.slider("TOP3 row threshold", 0.05, 0.60, 0.25, step=0.05)
    row_top2_rate_threshold = st.slider("TOP2 row threshold", 0.05, 0.80, 0.35, step=0.05)
    row_low_top2_rate_threshold = st.slider("TOP1-only if Top2 rate below", 0.00, 0.50, 0.35, step=0.05)
    row_top1_rate_threshold = st.slider("TOP1-only row threshold", 0.30, 1.00, 0.70, step=0.05)

    st.header("Universal Score Weights")
    w_model = st.slider("Universal weight: model confidence", 0.0, 1.0, 0.45, step=0.05)
    w_row = st.slider("Universal weight: row percentile", 0.0, 1.0, 0.25, step=0.05)
    w_density = st.slider("Universal weight: stream density", 0.0, 1.0, 0.20, step=0.05)
    w_outlier = st.slider("Universal penalty: miss/outlier risk", 0.0, 1.0, 0.10, step=0.05)

    truth_min_rate = st.slider("Truth-mined min rate", 0.50, 0.95, 0.76, step=0.01)
    truth_min_support = st.slider("Truth-mined min support", 1, 25, 5)

tab_daily, tab_straight, tab_straight_backtest, tab_lab, tab_stream, tab_help = st.tabs(["Daily Prediction", "Straight Plays", "Straight Backtest", "Walk-forward Lab", "Stream / Row Tests", "Notes"])

if hist_file is None:
    st.info("Upload FULL HISTORY to begin.")
    st.stop()

try:
    hist = normalize_history(load_upload(hist_file))
    if last24_file is not None:
        last24 = normalize_history(load_upload(last24_file))
        hist = pd.concat([hist, last24], ignore_index=True)
        hist = hist.drop_duplicates(subset=["Date", "StreamKey", "Result"]).sort_values(["StreamKey", "Date"]).reset_index(drop=True)
except Exception as e:
    st.error(f"Could not load history: {e}")
    st.stop()

try:
    truth = ensure_truth(load_upload(truth_file)) if truth_file is not None else None
    lib = ensure_library(pd.read_csv(lib_file, dtype=str)) if lib_file is not None else None
except Exception as e:
    st.error(f"Could not load truth/library file: {e}")
    st.stop()

stream_diag, kept_streams = stream_reduction(hist, window_days, prune_pct)
stream_diag = assign_stream_tiers(stream_diag)
stream_diag = add_stream_selection_flags(stream_diag, recommended_top_n, row_pct_cutoff)
rules = pd.DataFrame(columns=["pair", "trait_stack", "winner_member", "winner_rate", "support", "pair_gap", "stack_size"])
new_truth_rules = pd.DataFrame(columns=rules.columns)
if lib is not None:
    rules = pd.concat([rules, lib], ignore_index=True)
if truth is not None:
    truth = enrich_seed_features(truth, "seed")
    new_truth_rules = deep_mine_separators(truth, truth_min_rate, truth_min_support)
    if not new_truth_rules.empty:
        rules = pd.concat([rules, new_truth_rules], ignore_index=True)


# ================================
# v62 LOCKED EXECUTION LAYER
# Full ranking remains visible; actual default playable output is locked to TOP_20_STREAM_ROWS.
# ================================

V62_LOCKED_STREAM_RANK_MAX = 20
V62_LOCKED_ROW_PERCENTILE_MAX = 25.0
V62_MIN_ROW_CAPTURE_PCT = 75.0

def v62_row_capture_lookup() -> Dict[int, float]:
    df = st.session_state.get("row_single_perf")
    out = {}
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return out
    if "SingleRow" not in df.columns or "capture_pct" not in df.columns:
        return out
    for _, r in df.iterrows():
        try:
            out[int(float(r["SingleRow"]))] = float(r["capture_pct"])
        except Exception:
            pass
    return out

def v62_apply_locked_execution_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["v62_StreamRank_OK"] = pd.to_numeric(out.get("StreamRank", 9999), errors="coerce").fillna(9999).le(V62_LOCKED_STREAM_RANK_MAX)
    out["v62_RowPct_Audit_OK"] = pd.to_numeric(out.get("RowPercentile", 9999), errors="coerce").fillna(9999).le(V62_LOCKED_ROW_PERCENTILE_MAX)

    row_capture = v62_row_capture_lookup()
    if row_capture:
        out["v62_RowCapturePct"] = pd.to_numeric(out.get("SingleRow", -1), errors="coerce").fillna(-1).astype(int).map(row_capture).fillna(0.0)
        out["v62_RowCapture_Audit_OK"] = out["v62_RowCapturePct"].ge(V62_MIN_ROW_CAPTURE_PCT)
    else:
        out["v62_RowCapturePct"] = None
        out["v62_RowCapture_Audit_OK"] = True

    # Verified high-capture execution method: TOP_20_STREAM_ROWS.
    out["v62_DefaultPlayable"] = out["v62_StreamRank_OK"].astype(int)
    out["v62_ExecutionReason"] = out["v62_DefaultPlayable"].map({
        1: "PLAY:TOP_20_STREAM_ROWS_LOCKED",
        0: "REVIEW_ONLY:OUTSIDE_TOP_20_STREAM_ROWS"
    })
    return out

def v62_default_play_plan(df: pd.DataFrame) -> pd.DataFrame:
    marked = v62_apply_locked_execution_filter(df)
    if marked.empty:
        return marked
    return marked[marked["v62_DefaultPlayable"].astype(int).eq(1)].copy()

def v62_execution_summary(full_df: pd.DataFrame, play_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "Build": BUILD_MARKER,
        "ExecutionMode": "LOCKED_TOP_20_STREAM_ROWS",
        "FullRowsVisible": 0 if full_df is None else len(full_df),
        "DefaultPlayableRows": 0 if play_df is None else len(play_df),
        "LockedStreamRankMax": V62_LOCKED_STREAM_RANK_MAX,
        "ReferenceMethod": "TOP_20_STREAM_ROWS",
        "ReferenceCapturePct": 94.62,
        "ReferenceMisses": 5,
        "ReferencePlaysPerCapture": 2.466,
    }])


with tab_daily:
    st.subheader("Daily 025 prediction playlist")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("History rows", f"{len(hist):,}")
    c2.metric("Streams", hist["StreamKey"].nunique())
    c3.metric("Kept streams", len(kept_streams))
    c4.metric("Total active rules", len(rules))
    st.caption(f"Recommended stream marker: Top-N={recommended_top_n}, Row percentile cutoff={row_pct_cutoff}%. All streams still remain visible.")

    if lib is None:
        st.warning("Upload promoted separator library for v22/v31-style member scoring.")
    if truth is None:
        st.warning("Upload FULL TRUTH file if you want live truth-mined intelligence layer active.")

    if st.button("Run Daily Prediction", type="primary", use_container_width=True, key="daily_run"):
        status = st.status("Running daily prediction...", expanded=True)
        prog = st.progress(0, text="Preparing stream rows")
        live_rows = build_live_seed_rows(hist, kept_streams)
        prog.progress(35, text="Scoring members")
        scored = []
        for _, row in live_rows.iterrows():
            pick = v22_pick(row, rules, trait_weight, min_margin, enable_margin_swap, top2_zone_ratio, top2_zone_margin, top2_risk_threshold) if not rules.empty else {
                "PredictedMember": "0025", "Top2_pred": "0225", "Margin": 0.0, "Fired_Rules": "",
                "GATED_TOP3": 0, "score_0025": 0, "score_0225": 0, "score_0255": 0
            }
            scored.append({**row.to_dict(), **pick})
        playlist = pd.DataFrame(scored)
        if not playlist.empty:
            merge_cols = [c for c in [
                "StreamKey", "hit_density", "rank", "StreamRank", "kept", "RowPercentile",
                "SingleRow", "RowBucket10", "StreamTier", "RecommendedByTopN",
                "RecommendedByRowPct", "RecommendedStream"
            ] if c in stream_diag.columns]
            playlist = playlist.merge(stream_diag[merge_cols], on="StreamKey", how="left")
            playlist = add_rare_top3_and_universal_scores(
                playlist, enable_top3_rescue, top3_rescue_min_score,
                w_model, w_row, w_density, w_outlier
            )
            row_model_df = build_row_play_model(
                st.session_state.get("row_single_perf"),
                row_model_min_hits,
                row_top3_rate_threshold,
                row_top2_rate_threshold,
                row_top1_rate_threshold,
                row_low_top2_rate_threshold,
            )
            st.session_state["row_play_model"] = row_model_df
            playlist = apply_row_playtype_model(playlist, row_model_df, enable_row_playtype_model)
            playlist = apply_play_recommendation(playlist, play_policy)
            playlist = playlist.sort_values(["RecommendedStream", "UniversalFamilyScore", "hit_density", "Margin"], ascending=[False, False, False, False]).reset_index(drop=True)
            playlist["PlaylistRank"] = playlist.index + 1
            playlist["margin_percentile"] = playlist["Margin"].rank(pct=True, method="average").round(4)
            front_cols = [
                "PlaylistRank", "RecommendedStream", "StreamRank", "RowPercentile", "SingleRow", "StreamTier",
                "StreamKey", "State", "Game", "LastResult", "UniversalFamilyScore", "PlayType", "RecommendedPlay",
                "Top1_Display", "Top2_Display", "Top3_Display", "PredictedMember", "Top2_pred", "ThirdMember",
                "Top3Rescue", "Top3RescueScore", "Top3RescueReasons",
                "RowPlayType", "RowPlayTypeReason", "RowModelHits", "RowTop1Rate", "RowTop2Rate", "RowTop3Rate", "RowLowTop2Threshold",
                "Margin", "Top2ToTop1Ratio",
                "Top2Zone", "Top2RiskScore", "Top2Decision", "Top2RiskReasons", "DecisionMode",
                "score_0025", "score_0225", "score_0255", "ModelConfidenceScore",
                "RowStrengthScore", "FamilyHitDensityScore", "MissOutlierRiskScore", "hit_density", "Fired_Rules"
            ]
            playlist = playlist[[c for c in front_cols if c in playlist.columns] + [c for c in playlist.columns if c not in front_cols]]
        st.session_state["daily_playlist"] = playlist
        st.session_state["daily_playlist_full_visible"] = st.session_state["daily_playlist"].copy()
        st.session_state["daily_playlist_execution_marked"] = v62_apply_locked_execution_filter(st.session_state["daily_playlist"])
        st.session_state["daily_playlist_playable"] = v62_default_play_plan(st.session_state["daily_playlist"])
        st.session_state["v62_execution_summary"] = v62_execution_summary(st.session_state["daily_playlist"], st.session_state["daily_playlist_playable"])
        prog.progress(100, text="Done")
        status.update(label="Daily prediction complete", state="complete", expanded=False)

    if st.session_state["daily_playlist"] is not None:
        playlist = st.session_state["daily_playlist"]
        st.subheader("Visible Daily Playlist")
        if playlist.empty:
            st.warning("Daily playlist is empty after current filters.")
        else:
            st.dataframe(playlist, use_container_width=True, hide_index=True)
        a, b = st.columns(2)
        with a:
            st.download_button("Download Daily Playlist CSV", playlist.to_csv(index=False).encode("utf-8"), "daily_playlist__core025_northern_lights_v62_locked_v47_box.csv", "text/csv", use_container_width=True, on_click="ignore",
            key="dl_v62_0_download_daily_playlist_csv"
        )
        with b:
            st.download_button("Download Daily Playlist TXT", playlist.to_csv(index=False).encode("utf-8"), "daily_playlist__core025_northern_lights_v62_locked_v47_box.txt", "text/plain", use_container_width=True, on_click="ignore",
            key="dl_v62_1_download_daily_playlist_txt"
        )

    st.subheader("Stream reduction")
    st.dataframe(stream_diag, use_container_width=True, hide_index=True)



def v62_truth_event_playlist_from_row(row: pd.Series) -> pd.DataFrame:
    """Create a one-row playlist from a lab/per-event truth row using locked member outputs."""
    r = row.copy()
    if "PlaylistRank" not in r.index:
        r["PlaylistRank"] = r.get("rank", r.get("StreamRank", ""))
    if "LastResult" not in r.index:
        r["LastResult"] = r.get("seed", "")
    if "PlayType" not in r.index:
        r["PlayType"] = r.get("RowPlayType", "")
    if "RecommendedPlay" not in r.index:
        parts = []
        for col in ["PredictedMember", "Top2_pred", "ThirdMember"]:
            val = normalize_member(r.get(col, ""))
            if val in MEMBERS and val not in parts:
                parts.append(val)
        if "TOP1_ONLY" in str(r.get("PlayType", "")).upper():
            parts = parts[:1]
        elif "TOP1_TOP2_TOP3" in str(r.get("PlayType", "")).upper():
            parts = parts[:3]
        else:
            parts = parts[:2]
        r["RecommendedPlay"] = " + ".join(parts)
    return pd.DataFrame([r.to_dict()])

def v62_run_straight_backtest(lab_df: pd.DataFrame, hist: pd.DataFrame, recommended_only: bool = True, max_events: int = 0, straight_depth: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if lab_df is None or lab_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    events = lab_df.copy()
    if recommended_only:
        if "StreamRank" in events.columns:
            events = events[pd.to_numeric(events["StreamRank"], errors="coerce").fillna(9999).le(V62_LOCKED_STREAM_RANK_MAX)].copy()
        elif "RecommendedStream" in events.columns:
            events = events[pd.to_numeric(events["RecommendedStream"], errors="coerce").fillna(0).astype(int).eq(1)].copy()

    if max_events and int(max_events) > 0:
        events = events.head(int(max_events)).copy()

    rows = []
    for idx, ev in events.iterrows():
        true_result = extract_pick4_digits(ev.get("Result", ""))
        true_member = normalize_member(ev.get("TrueMember", result_to_core025_member(true_result)))
        playlist_one = v62_truth_event_playlist_from_row(ev)

        ranked, _summary = v62_generate_downstream_straights(
            playlist_one,
            hist,
            recommended_only=False,
            straight_depth=max(24, int(straight_depth))
        )

        if ranked is None or ranked.empty:
            rows.append({
                "EventIndex": idx,
                "StreamKey": ev.get("StreamKey", ""),
                "Date": ev.get("Date", ""),
                "seed": ev.get("seed", ev.get("LastResult", "")),
                "Result": true_result,
                "TrueMember": true_member,
                "PredictedMember": ev.get("PredictedMember", ""),
                "Top2_pred": ev.get("Top2_pred", ""),
                "ThirdMember": ev.get("ThirdMember", ""),
                "PlayType": ev.get("PlayType", ev.get("RowPlayType", "")),
                "StraightHitRank": None,
                "StraightCapturedAtDepth": 0,
                "StraightHit": 0,
                "Reason": "NO_RANKED_STRAIGHTS"
            })
            continue

        ranked["StraightPermutation"] = ranked["StraightPermutation"].astype(str).str.zfill(4)
        hit_rows = ranked[ranked["StraightPermutation"].eq(str(true_result).zfill(4))].copy()
        if hit_rows.empty:
            hit_rank = None
            hit_score = None
            hit_member = None
            hit = 0
            reason = "TRUE_STRAIGHT_NOT_IN_ELIGIBLE_MEMBER_SET"
        else:
            hr = hit_rows.sort_values("StreamStraightRank").iloc[0]
            hit_rank = int(hr["StreamStraightRank"])
            hit_score = hr.get("StraightConfidenceScore", None)
            hit_member = hr.get("StraightMember", "")
            hit = 1
            reason = "HIT_IN_RANKED_LIST"

        rows.append({
            "EventIndex": idx,
            "StreamKey": ev.get("StreamKey", ""),
            "Date": ev.get("Date", ""),
            "seed": ev.get("seed", ev.get("LastResult", "")),
            "Result": true_result,
            "TrueMember": true_member,
            "PredictedMember": ev.get("PredictedMember", ""),
            "Top2_pred": ev.get("Top2_pred", ""),
            "ThirdMember": ev.get("ThirdMember", ""),
            "PlayType": ev.get("PlayType", ev.get("RowPlayType", "")),
            "SingleRow": ev.get("SingleRow", ""),
            "StreamRank": ev.get("StreamRank", ""),
            "RowPercentile": ev.get("RowPercentile", ""),
            "RecommendedStream": ev.get("RecommendedStream", ""),
            "Top1_Correct": ev.get("Top1_Correct", ""),
            "Needed_Top2": ev.get("Needed_Top2", ""),
            "Miss": ev.get("Miss", ""),
            "StraightHit": hit,
            "StraightHitRank": hit_rank,
            "StraightHitMember": hit_member,
            "StraightHitScore": hit_score,
            "StraightCapturedAtDepth": 1 if (hit_rank is not None and int(hit_rank) <= int(straight_depth)) else 0,
            "Reason": reason,
        })

    per_event = pd.DataFrame(rows)

    total = len(per_event)
    ranked_hits = int(per_event["StraightHit"].sum()) if total else 0
    captured_depth = int(per_event["StraightCapturedAtDepth"].sum()) if total else 0
    summary = pd.DataFrame([{
        "Build": BUILD_MARKER,
        "events_tested": total,
        "straight_ranked_hits": ranked_hits,
        "straight_ranked_hit_pct": round((ranked_hits / total * 100) if total else 0, 2),
        "straight_depth": int(straight_depth),
        "straight_captured_at_depth": captured_depth,
        "straight_capture_at_depth_pct": round((captured_depth / total * 100) if total else 0, 2),
        "recommended_only": recommended_only,
        "plays_per_event": int(straight_depth),
        "estimated_straight_plays": int(total * int(straight_depth)),
    }])

    depth_rows = []
    if total:
        for d in range(1, 25):
            cnt = int(per_event["StraightHitRank"].apply(lambda x: pd.notna(x) and int(x) <= d).sum())
            depth_rows.append({
                "Depth": d,
                "HitsAtOrAboveDepth": cnt,
                "HitPct": round(cnt / total * 100, 2),
                "EstimatedPlays": total * d,
                "Events": total,
            })
    depth_df = pd.DataFrame(depth_rows)
    return per_event, summary, depth_df



    st.markdown("### v62 Locked Execution Play Plan")
    st.caption("Actual default playable subset: StreamRank ≤ 20 (TOP_20_STREAM_ROWS). Full stream list remains visible for audit.")
    play_df_v62 = st.session_state.get("daily_playlist_playable")
    marked_df_v62 = st.session_state.get("daily_playlist_execution_marked")
    summary_v62 = st.session_state.get("v62_execution_summary")

    if summary_v62 is not None and isinstance(summary_v62, pd.DataFrame) and not summary_v62.empty:
        st.dataframe(summary_v62, use_container_width=True, hide_index=True)

    if play_df_v62 is not None and isinstance(play_df_v62, pd.DataFrame) and not play_df_v62.empty:
        st.success(f"v62 locked play plan active: {len(play_df_v62)} playable rows. Full list remains visible separately.")
        st.dataframe(play_df_v62, use_container_width=True, hide_index=True)
        st.download_button(
            "Download v62 LOCKED PLAY PLAN CSV",
            play_df_v62.to_csv(index=False).encode("utf-8"),
            "daily_locked_play_plan__core025_northern_lights_v62.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
            key="dl_v62_locked_play_plan",
        )
    else:
        st.info("Run Daily Prediction to generate the locked v62 play plan.")

    if marked_df_v62 is not None and isinstance(marked_df_v62, pd.DataFrame) and not marked_df_v62.empty:
        with st.expander("Full ranked list with v62 playable/review-only marks", expanded=False):
            st.dataframe(marked_df_v62, use_container_width=True, hide_index=True)
        st.download_button(
            "Download v62 FULL RANKED LIST WITH EXECUTION MARKS CSV",
            marked_df_v62.to_csv(index=False).encode("utf-8"),
            "daily_full_ranked_with_v62_execution_marks__core025_northern_lights_v62.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
            key="dl_v62_full_execution_marks",
        )

with tab_straight:
    st.subheader("v62 Downstream Straight Plays")
    st.caption("Locked rule: this tab uses the existing daily member picks only. It does not rescore or change PredictedMember, Top2_pred, ThirdMember, PlayType, or RecommendedPlay.")
    straight_depth = st.number_input("Recommended straights per stream", min_value=1, max_value=6, value=2, step=1)
    recommended_only = st.checkbox("Use v62 locked playable streams only", value=True)

    if st.button("Run v62 Straight Plays", type="primary", use_container_width=True, key="v62_run_straights"):
        playlist = st.session_state.get("daily_playlist_playable") if recommended_only else st.session_state.get("daily_playlist")
        if playlist is None or playlist.empty:
            st.warning("Run Daily Prediction first. Straight Plays needs the locked v47/v45 box playlist.")
        else:
            ranked, summary = v62_generate_downstream_straights(playlist, hist, recommended_only=recommended_only, straight_depth=int(straight_depth))
            st.session_state["straight_ranked"] = ranked
            st.session_state["straight_summary"] = summary
            st.session_state["straight_recommended"] = ranked[ranked["RecommendedStraight"].eq(1)].copy() if not ranked.empty else pd.DataFrame()

    if st.session_state.get("straight_recommended") is not None:
        rec = st.session_state.get("straight_recommended")
        summ = st.session_state.get("straight_summary")
        ranked = st.session_state.get("straight_ranked")
        if rec is not None and not rec.empty:
            st.markdown("### Recommended Straight Plays")
            st.dataframe(rec, use_container_width=True, hide_index=True)
            st.download_button(
                "Download v62 Recommended Straights CSV",
                rec.to_csv(index=False).encode("utf-8"),
                "straight_recommended_only__core025_northern_lights_v62.csv",
                "text/csv",
                use_container_width=True,
                on_click="ignore",
                key="dl_v62_straight_recommended",
            )
        if summ is not None and not summ.empty:
            st.markdown("### Straight Summary")
            st.dataframe(summ, use_container_width=True, hide_index=True)
            st.download_button(
                "Download v62 Straight Summary CSV",
                summ.to_csv(index=False).encode("utf-8"),
                "straight_summary__core025_northern_lights_v62.csv",
                "text/csv",
                use_container_width=True,
                on_click="ignore",
                key="dl_v62_straight_summary",
            )
        if ranked is not None and not ranked.empty:
            with st.expander("Full ranked straight candidates", expanded=False):
                st.dataframe(ranked, use_container_width=True, hide_index=True)
            st.download_button(
                "Download v62 Full Ranked Straights CSV",
                ranked.to_csv(index=False).encode("utf-8"),
                "straight_ranked__core025_northern_lights_v62.csv",
                "text/csv",
                use_container_width=True,
                on_click="ignore",
                key="dl_v62_straight_ranked",
            )
    else:
        st.info("Run Daily Prediction first, then run this tab.")




with tab_straight_backtest:
    st.subheader("v62 Straight Play Backtest")
    st.caption("Tests the downstream straight layer against lab events. The locked v47/v45 box/member outputs are used as-is.")

    bt_depth = st.number_input("Backtest straight depth", min_value=1, max_value=24, value=2, step=1, key="v62_bt_depth")
    bt_recommended_only = st.checkbox("Backtest recommended streams only", value=True, key="v62_bt_recommended_only")
    bt_max_events = st.number_input("Max events to process (0 = all)", min_value=0, max_value=10000, value=0, step=25, key="v62_bt_max_events")

    st.info("Run Walk-forward Lab first, then run this tab. It uses st.session_state['lab_results'] from the locked box/member engine.")

    if st.button("Run v62 Straight Backtest", type="primary", use_container_width=True, key="v62_run_straight_backtest"):
        lab_df = st.session_state.get("lab_results")
        if lab_df is None or lab_df.empty:
            st.warning("Run the Walk-forward Lab first. Straight Backtest needs lab_results.")
        else:
            per_event, summary, depth_df = v62_run_straight_backtest(
                lab_df,
                hist,
                recommended_only=bt_recommended_only,
                max_events=int(bt_max_events),
                straight_depth=int(bt_depth),
            )
            st.session_state["straight_backtest_events"] = per_event
            st.session_state["straight_backtest_summary"] = summary
            st.session_state["straight_backtest_depth"] = depth_df

    bt_summary = st.session_state.get("straight_backtest_summary")
    bt_events = st.session_state.get("straight_backtest_events")
    bt_depth_df = st.session_state.get("straight_backtest_depth")

    if bt_summary is not None and not bt_summary.empty:
        st.markdown("### Straight Backtest Summary")
        st.dataframe(bt_summary, use_container_width=True, hide_index=True)
        st.download_button(
            "Download v62 Straight Backtest Summary CSV",
            bt_summary.to_csv(index=False).encode("utf-8"),
            "straight_backtest_summary__core025_northern_lights_v62.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
            key="dl_v62_straight_bt_summary",
        )

    if bt_depth_df is not None and not bt_depth_df.empty:
        st.markdown("### Depth Curve")
        st.dataframe(bt_depth_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download v62 Straight Depth Curve CSV",
            bt_depth_df.to_csv(index=False).encode("utf-8"),
            "straight_backtest_depth_curve__core025_northern_lights_v62.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
            key="dl_v62_straight_bt_depth",
        )

    if bt_events is not None and not bt_events.empty:
        st.markdown("### Per-Event Straight Backtest")
        st.dataframe(bt_events, use_container_width=True, hide_index=True)
        st.download_button(
            "Download v62 Straight Backtest Per-Event CSV",
            bt_events.to_csv(index=False).encode("utf-8"),
            "straight_backtest_per_event__core025_northern_lights_v62.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
            key="dl_v62_straight_bt_events",
        )



with tab_lab:
    st.subheader("Walk-forward lab on uploaded history")
    if st.button("Run Walk-forward Lab", type="primary", use_container_width=True, key="lab_run"):
        status = st.status("Running walk-forward lab...", expanded=True)
        prog = st.progress(0, text="Building event rows")
        lab_rows = build_lab_event_rows(hist)
        prog.progress(35, text="Applying stream reduction")
        lab_rows = lab_rows[lab_rows["StreamKey"].isin(kept_streams)].copy()
        scored = []
        prog.progress(60, text="Scoring rows")
        for _, row in lab_rows.iterrows():
            pick = v22_pick(row, rules, trait_weight, min_margin, enable_margin_swap, top2_zone_ratio, top2_zone_margin, top2_risk_threshold) if not rules.empty else {
                "PredictedMember": "0025", "Top2_pred": "0225", "Margin": 0.0, "Fired_Rules": "",
                "GATED_TOP3": 0, "score_0025": 0, "score_0225": 0, "score_0255": 0
            }
            top1 = int(pick["PredictedMember"] == row["TrueMember"])
            top2_needed = int(top1 == 0 and pick["Top2_pred"] == row["TrueMember"])
            miss = int(top1 == 0 and top2_needed == 0)
            scored.append({**row.to_dict(), **pick, "Top1_Correct": top1, "Needed_Top2": top2_needed, "Miss": miss})
        res = pd.DataFrame(scored)
        if not res.empty:
            merge_cols = [c for c in [
                "StreamKey", "hit_density", "rank", "StreamRank", "kept", "RowPercentile",
                "SingleRow", "RowBucket10", "StreamTier", "RecommendedByTopN",
                "RecommendedByRowPct", "RecommendedStream"
            ] if c in stream_diag.columns]
            res = res.merge(stream_diag[merge_cols], on="StreamKey", how="left")
            res = add_rare_top3_and_universal_scores(
                res, enable_top3_rescue, top3_rescue_min_score,
                w_model, w_row, w_density, w_outlier
            )
            row_model_df = build_row_play_model(
                st.session_state.get("row_single_perf"),
                row_model_min_hits,
                row_top3_rate_threshold,
                row_top2_rate_threshold,
                row_top1_rate_threshold,
                row_low_top2_rate_threshold,
            )
            st.session_state["row_play_model"] = row_model_df
            res = apply_row_playtype_model(res, row_model_df, enable_row_playtype_model)
            res = apply_play_recommendation(res, play_policy)
        total = len(res)
        top1 = int(res["Top1_Correct"].sum()) if total else 0
        needed = int(res["Needed_Top2"].sum()) if total else 0
        miss = int(res["Miss"].sum()) if total else 0
        capture = (top1 + needed) / total * 100 if total else 0.0

        operational_top2_capture = int(((res["Needed_Top2"] == 1) & (res["PlayType"].isin(["TOP1_TOP2", "TOP1_TOP2_TOP3"]))).sum()) if total and "PlayType" in res.columns else 0
        operational_top3_capture = int(((res["Miss"] == 1) & (res["PlayType"] == "TOP1_TOP2_TOP3")).sum()) if total and "PlayType" in res.columns else 0
        operational_captured = top1 + operational_top2_capture + operational_top3_capture
        operational_capture_pct = (operational_captured / total * 100) if total else 0.0
        operational_plays = 0
        if total and "PlayType" in res.columns:
            operational_plays = int((res["PlayType"] == "TOP1_ONLY").sum() + 2*(res["PlayType"] == "TOP1_TOP2").sum() + 3*(res["PlayType"] == "TOP1_TOP2_TOP3").sum())
        summary = pd.DataFrame([{
            "total_rows": total,
            "top1": top1,
            "top1_pct": round((top1 / total * 100), 2) if total else 0.0,
            "needed_top2": needed,
            "top2_burden_pct": round((needed / total * 100), 2) if total else 0.0,
            "miss": miss,
            "capture_pct": round(capture, 2),
            "operational_captured": operational_captured,
            "operational_capture_pct": round(operational_capture_pct, 2),
            "operational_plays": operational_plays,
            "operational_plays_per_capture": round(operational_plays / operational_captured, 3) if operational_captured else None,
            "top3_rescue_rows": int(res["Top3Rescue"].sum()) if total and "Top3Rescue" in res.columns else 0,
            "top3_rescue_captured_misses": operational_top3_capture,
            "gated_top3": int(res["GATED_TOP3"].sum()) if total else 0,
            "margin_swaps": int(res["MarginSwap"].sum()) if total and "MarginSwap" in res.columns else 0,
            "top2_zone_rows": int(res["Top2Zone"].sum()) if total and "Top2Zone" in res.columns else 0,
            "top2_required_rows": int((res["Top2Decision"] == "TOP2_REQUIRED").sum()) if total and "Top2Decision" in res.columns else 0,
            "top2_safe_rows": int((res["Top2Decision"] == "TOP1_SAFE").sum()) if total and "Top2Decision" in res.columns else 0,
            "kept_streams": len(kept_streams),
            "active_rules": len(rules),
            "truth_mined_rules": len(new_truth_rules),
            "decision_layer": "strict_score_ranking_no_third_hijack",
            "margin_swap_enabled": enable_margin_swap,
            "play_policy": play_policy,
            "recommended_stream_rows": int(res["RecommendedStream"].sum()) if total and "RecommendedStream" in res.columns else 0,
            "row_playtype_model_enabled": enable_row_playtype_model,
            "row_model_min_hits": row_model_min_hits,
            "row_top3_rate_threshold": row_top3_rate_threshold,
            "row_top2_rate_threshold": row_top2_rate_threshold,
            "row_low_top2_rate_threshold": row_low_top2_rate_threshold,
            "row_top1_rate_threshold": row_top1_rate_threshold,
            "row_model_top1_only_rows": int((res["PlayType"] == "TOP1_ONLY").sum()) if total and "PlayType" in res.columns else 0,
            "row_model_top1_top2_rows": int((res["PlayType"] == "TOP1_TOP2").sum()) if total and "PlayType" in res.columns else 0,
            "row_model_top1_top2_top3_rows": int((res["PlayType"] == "TOP1_TOP2_TOP3").sum()) if total and "PlayType" in res.columns else 0,
        }])
        method_tests, single_row_perf = build_stream_method_tests(res, play_policy)
        pct = res.copy()
        if not pct.empty:
            pct["margin_percentile"] = pct["Margin"].rank(pct=True, method="average").round(4)
        st.session_state["lab_results"] = res
        st.session_state["lab_summary"] = summary
        st.session_state["percentile_df"] = pct
        st.session_state["stream_method_tests"] = method_tests
        st.session_state["row_single_perf"] = single_row_perf
        prog.progress(100, text="Done")
        status.update(label="Walk-forward lab complete", state="complete", expanded=False)

    if st.session_state["lab_summary"] is not None:
        summary = st.session_state["lab_summary"]
        res = st.session_state["lab_results"]
        pct = st.session_state["percentile_df"]
        st.dataframe(summary, use_container_width=True, hide_index=True)
        if not res.empty:
            front_cols = [
                "StreamKey", "StreamRank", "RowPercentile", "SingleRow", "StreamTier", "RecommendedStream",
                "Date", "seed", "Result", "TrueMember", "UniversalFamilyScore", "PlayType", "RecommendedPlay",
                "PredictedMember", "Top2_pred", "ThirdMember", "Top3Rescue", "Top3RescueScore",
                "Top3RescueReasons", "RowPlayType", "RowPlayTypeReason", "RowModelHits",
                "RowTop1Rate", "RowTop2Rate", "RowTop3Rate",
                "Top1_Correct", "Needed_Top2", "Miss",
                "Margin", "Top2ToTop1Ratio", "Top2Zone", "Top2RiskScore", "Top2Decision",
                "Top2RiskReasons", "DecisionMode", "MarginSwap",
                "score_0025", "score_0225", "score_0255", "Fired_Rules"
            ]
            res_view = res[[c for c in front_cols if c in res.columns] + [c for c in res.columns if c not in front_cols]]
        else:
            res_view = res
        st.dataframe(res_view, use_container_width=True, hide_index=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button("Download Lab Summary CSV", summary.to_csv(index=False).encode("utf-8"), "lab_summary__core025_northern_lights_v62_locked_v47_box.csv", "text/csv", use_container_width=True, on_click="ignore",
            key="dl_v62_2_download_lab_summary_csv"
        )
            st.download_button("Download Lab Summary TXT", summary.to_csv(index=False).encode("utf-8"), "lab_summary__core025_northern_lights_v62_locked_v47_box.txt", "text/plain", use_container_width=True, on_click="ignore",
            key="dl_v62_3_download_lab_summary_txt"
        )
        with c2:
            st.download_button("Download Lab Per Event CSV", res.to_csv(index=False).encode("utf-8"), "lab_per_event__core025_northern_lights_v62_locked_v47_box.csv", "text/csv", use_container_width=True, on_click="ignore",
            key="dl_v62_4_download_lab_per_event_csv"
        )
            st.download_button("Download Lab Per Event TXT", res.to_csv(index=False).encode("utf-8"), "lab_per_event__core025_northern_lights_v62_locked_v47_box.txt", "text/plain", use_container_width=True, on_click="ignore",
            key="dl_v62_5_download_lab_per_event_txt"
        )
        with c3:
            st.download_button("Download Percentile CSV", pct.to_csv(index=False).encode("utf-8"), "percentile_list__core025_northern_lights_v62_locked_v47_box.csv", "text/csv", use_container_width=True, on_click="ignore",
            key="dl_v62_6_download_percentile_csv"
        )
            st.download_button("Download Stream Method Tests CSV", st.session_state["stream_method_tests"].to_csv(index=False).encode("utf-8") if st.session_state["stream_method_tests"] is not None else b"", "stream_method_tests__core025_northern_lights_v62_locked_v47_box.csv", "text/csv", use_container_width=True, on_click="ignore",
            key="dl_v62_7_download_stream_method_tests_csv"
        )
        with c4:
            st.download_button("Download Stream Reduction CSV", stream_diag.to_csv(index=False).encode("utf-8"), "stream_reduction__core025_northern_lights_v62_locked_v47_box.csv", "text/csv", use_container_width=True, on_click="ignore",
            key="dl_v62_8_download_stream_reduction_csv"
        )
            st.download_button("Download Single Row Perf CSV", st.session_state["row_single_perf"].to_csv(index=False).encode("utf-8") if st.session_state["row_single_perf"] is not None else b"", "single_row_performance__core025_northern_lights_v62_locked_v47_box.csv", "text/csv", use_container_width=True, on_click="ignore",
            key="dl_v62_9_download_single_row_perf_csv"
        )


with tab_stream:
    st.subheader("Stream / Row-Based Percentile Testing")
    st.caption("This tab keeps all streams visible but tests which row bands, single rows, and stream-selection methods historically performed best. It does not change the baseline unless you choose to use its output operationally.")

    st.markdown("### Current stream list with row-based percentile scoring")
    st.dataframe(stream_diag, use_container_width=True, hide_index=True)

    if st.session_state["stream_method_tests"] is not None:
        st.markdown("### Method comparison")
        st.dataframe(st.session_state["stream_method_tests"], use_container_width=True, hide_index=True)

    if st.session_state["row_single_perf"] is not None:
        st.markdown("### Single-row performance")
        st.dataframe(st.session_state["row_single_perf"], use_container_width=True, hide_index=True)

    if st.session_state["row_play_model"] is not None:
        st.markdown("### Row Play-Type Model")
        st.dataframe(st.session_state["row_play_model"], use_container_width=True, hide_index=True)

    st.markdown("### Stream / Row Test Downloads")
    stream_tests_ready = st.session_state["stream_method_tests"] is not None
    single_rows_ready = st.session_state["row_single_perf"] is not None
    pct_ready = st.session_state["percentile_df"] is not None

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Download Stream Method Tests CSV",
            st.session_state["stream_method_tests"].to_csv(index=False).encode("utf-8") if stream_tests_ready else b"",
            "stream_method_tests__core025_northern_lights_v62_locked_v47_box.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
            disabled=not stream_tests_ready,
            key="dl_v62_10_download_stream_method_tests_csv"
        )
        st.download_button(
            "Download Stream Method Tests TXT",
            st.session_state["stream_method_tests"].to_csv(index=False).encode("utf-8") if stream_tests_ready else b"",
            "stream_method_tests__core025_northern_lights_v62_locked_v47_box.txt",
            "text/plain",
            use_container_width=True,
            on_click="ignore",
            disabled=not stream_tests_ready,
            key="dl_v62_11_download_stream_method_tests_txt"
        )
    with d2:
        st.download_button(
            "Download Single Row Performance CSV",
            st.session_state["row_single_perf"].to_csv(index=False).encode("utf-8") if single_rows_ready else b"",
            "single_row_performance__core025_northern_lights_v62_locked_v47_box.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
            disabled=not single_rows_ready,
            key="dl_v62_12_download_single_row_performance_csv"
        )
        st.download_button(
            "Download Single Row Performance TXT",
            st.session_state["row_single_perf"].to_csv(index=False).encode("utf-8") if single_rows_ready else b"",
            "single_row_performance__core025_northern_lights_v62_locked_v47_box.txt",
            "text/plain",
            use_container_width=True,
            on_click="ignore",
            disabled=not single_rows_ready,
            key="dl_v62_13_download_single_row_performance_txt"
        )
    with d3:
        st.download_button(
            "Download Percentile List CSV",
            st.session_state["percentile_df"].to_csv(index=False).encode("utf-8") if pct_ready else b"",
            "percentile_list__core025_northern_lights_v62_locked_v47_box.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
            disabled=not pct_ready,
            key="dl_v62_14_download_percentile_list_csv"
        )
        st.download_button(
            "Download Percentile List TXT",
            st.session_state["percentile_df"].to_csv(index=False).encode("utf-8") if pct_ready else b"",
            "percentile_list__core025_northern_lights_v62_locked_v47_box.txt",
            "text/plain",
            use_container_width=True,
            on_click="ignore",
            disabled=not pct_ready,
            key="dl_v62_15_download_percentile_list_txt"
        )
    with d4:
        st.download_button(
            "Download Stream Reduction CSV",
            stream_diag.to_csv(index=False).encode("utf-8"),
            "stream_reduction__core025_northern_lights_v62_locked_v47_box.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
            key="dl_v62_16_download_stream_reduction_csv"
        )
        st.download_button(
            "Download Stream Reduction TXT",
            stream_diag.to_csv(index=False).encode("utf-8"),
            "stream_reduction__core025_northern_lights_v62_locked_v47_box.txt",
            "text/plain",
            use_container_width=True,
            on_click="ignore",
            key="dl_v62_17_download_stream_reduction_txt"
        )

    row_model_ready = st.session_state["row_play_model"] is not None
    d5, _ = st.columns([1, 3])
    with d5:
        st.download_button(
            "Download Row Play Model CSV",
            st.session_state["row_play_model"].to_csv(index=False).encode("utf-8") if row_model_ready else b"",
            "row_play_model__core025_northern_lights_v62_locked_v47_box.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
            disabled=not row_model_ready,
            key="dl_v62_stream_tab_row_play_model_csv",
        )

    if not stream_tests_ready:
        st.info("Stream Method, Single Row, and Percentile downloads unlock after Walk-forward Lab runs in this app session. Stream Reduction is downloadable immediately.")


with tab_help:
    st.markdown("""
### What this build keeps from Northern Lights
- Full history input
- Optional last 24h input
- Live daily prediction tab
- Walk-forward lab tab
- Stream reduction / low-density pruning
- Percentile list
- Downloadable outputs

### v44 decision-layer correction
- Keeps v34 scoring, promoted-library feature matching, full-truth mining, and stream reduction.
- Removes destructive third-choice GATED_TOP3 hijack.
- Top1 is the highest score and Top2 is the second-highest score.
- Optional cautious Top1/Top2 margin swap is OFF by default for clean testing.\n- Adds Top2-zone-gated deep risk scoring.\n- Deep risk traits are evaluated only after the first gate confirms the row is in a Top2 zone.

### What this build adds from the 75% v22/v31 path
- Promoted separator library scoring
- FULL TRUTH intelligence layer
- On-run truth-mined separators
- `winner_rate * trait_weight` scoring
- Strong `GATED_TOP3` logic
- 025-only member prediction: `0025`, `0225`, `0255`

### v44 stream / row scoring + rare Top3 rescue additions
- Adds PlayType and RecommendedPlay display logic.
- Full capture display can bold Top1+Top2, while risk-gate mode bolds Top2 only when TOP2_REQUIRED.
- Adds row-by-row percentile scoring: SingleRow, RowPercentile, RowBucket10, StreamTier.
- Adds a Stream / Row Tests tab for Top-N, percentile bands, tiers, and single-row historical performance.
- Adds Rare Top3 Rescue for miss/outlier rows before any Top2 waste trimming.
- Adds UniversalFamilyScore, scaled for future cross-family app comparison.
- Adds explicit Stream / Row Test download buttons for CSV and TXT exports.
- v44 corrects row play-type logic: low-sample rows default to TOP1_TOP2 instead of automatic Top3.
- v44 adds Top2 cost control: rows with Top2 rate below the selected threshold become TOP1_ONLY.
- Keeps unique download-button keys and Row Play Model export.

### v45 final optimized row play-type corrections
- Low-sample rows default to TOP1_TOP2, not automatic Top3.
- Top3 default threshold is 0.25 with minimum 4 historical hits.
- Top2 cost control default threshold is 0.35 in the UI, representing the optimized 0.33–0.35 band.
- All download filenames and build labels are v45.

### Required files
For daily prediction:
1. Full history
2. Promoted separator library
3. FULL TRUTH file recommended for intelligence layer
4. Last 24h optional

For lab:
Same files, then run Walk-forward Lab.
""")
