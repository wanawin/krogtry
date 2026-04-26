#!/usr/bin/env python3
# BUILD: core025_northern_lights__2026-04-25_v40_download_buttons_fixed

from __future__ import annotations

import io
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_northern_lights__2026-04-25_v40_download_buttons_fixed"
MEMBERS = ["0025", "0225", "0255"]

st.set_page_config(page_title="Core025 Northern Lights 025", layout="wide")
st.title("Core025 Northern Lights — 025 Live + Lab")
st.caption(BUILD_MARKER)
st.warning(
    "025-only merged build v40: play-type display, row-by-row percentile scoring, and stream-reduction method tests; v39 scoring preserved, "
    "Northern-Lights-style stream reduction, and v22/v31 truth-backed member engine."
)

for k in [
    "daily_playlist", "lab_results", "lab_summary", "stream_diag",
    "truth_rules", "loaded_meta", "percentile_df", "stream_method_tests", "row_single_perf"
]:
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

    if play_policy.startswith("Full capture"):
        out["PlayType"] = "TOP1_TOP2"
    elif play_policy.startswith("Top2 risk"):
        out["PlayType"] = out["Top2Decision"].apply(lambda x: "TOP1_TOP2" if str(x) == "TOP2_REQUIRED" else "TOP1_ONLY")
    else:
        out["PlayType"] = "TOP1_ONLY"

    # Rare Top3 rescue overrides play display only; it does not rewrite strict member scores.
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

    st.header("Universal Score Weights")
    w_model = st.slider("Universal weight: model confidence", 0.0, 1.0, 0.45, step=0.05)
    w_row = st.slider("Universal weight: row percentile", 0.0, 1.0, 0.25, step=0.05)
    w_density = st.slider("Universal weight: stream density", 0.0, 1.0, 0.20, step=0.05)
    w_outlier = st.slider("Universal penalty: miss/outlier risk", 0.0, 1.0, 0.10, step=0.05)

    truth_min_rate = st.slider("Truth-mined min rate", 0.50, 0.95, 0.76, step=0.01)
    truth_min_support = st.slider("Truth-mined min support", 1, 25, 5)

tab_daily, tab_lab, tab_stream, tab_help = st.tabs(["Daily Prediction", "Walk-forward Lab", "Stream / Row Tests", "Notes"])

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
            playlist = apply_play_recommendation(playlist, play_policy)
            playlist = playlist.sort_values(["RecommendedStream", "UniversalFamilyScore", "hit_density", "Margin"], ascending=[False, False, False, False]).reset_index(drop=True)
            playlist["PlaylistRank"] = playlist.index + 1
            playlist["margin_percentile"] = playlist["Margin"].rank(pct=True, method="average").round(4)
            front_cols = [
                "PlaylistRank", "RecommendedStream", "StreamRank", "RowPercentile", "SingleRow", "StreamTier",
                "StreamKey", "State", "Game", "LastResult", "UniversalFamilyScore", "PlayType", "RecommendedPlay",
                "Top1_Display", "Top2_Display", "Top3_Display", "PredictedMember", "Top2_pred", "ThirdMember",
                "Top3Rescue", "Top3RescueScore", "Top3RescueReasons", "Margin", "Top2ToTop1Ratio",
                "Top2Zone", "Top2RiskScore", "Top2Decision", "Top2RiskReasons", "DecisionMode",
                "score_0025", "score_0225", "score_0255", "ModelConfidenceScore",
                "RowStrengthScore", "FamilyHitDensityScore", "MissOutlierRiskScore", "hit_density", "Fired_Rules"
            ]
            playlist = playlist[[c for c in front_cols if c in playlist.columns] + [c for c in playlist.columns if c not in front_cols]]
        st.session_state["daily_playlist"] = playlist
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
            st.download_button("Download Daily Playlist CSV", playlist.to_csv(index=False).encode("utf-8"), "daily_playlist__core025_northern_lights_v40.csv", "text/csv", use_container_width=True, on_click="ignore")
        with b:
            st.download_button("Download Daily Playlist TXT", playlist.to_csv(index=False).encode("utf-8"), "daily_playlist__core025_northern_lights_v40.txt", "text/plain", use_container_width=True, on_click="ignore")

    st.subheader("Stream reduction")
    st.dataframe(stream_diag, use_container_width=True, hide_index=True)

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
                "Top3RescueReasons", "Top1_Correct", "Needed_Top2", "Miss",
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
            st.download_button("Download Lab Summary CSV", summary.to_csv(index=False).encode("utf-8"), "lab_summary__core025_northern_lights_v40.csv", "text/csv", use_container_width=True, on_click="ignore")
            st.download_button("Download Lab Summary TXT", summary.to_csv(index=False).encode("utf-8"), "lab_summary__core025_northern_lights_v40.txt", "text/plain", use_container_width=True, on_click="ignore")
        with c2:
            st.download_button("Download Lab Per Event CSV", res.to_csv(index=False).encode("utf-8"), "lab_per_event__core025_northern_lights_v40.csv", "text/csv", use_container_width=True, on_click="ignore")
            st.download_button("Download Lab Per Event TXT", res.to_csv(index=False).encode("utf-8"), "lab_per_event__core025_northern_lights_v40.txt", "text/plain", use_container_width=True, on_click="ignore")
        with c3:
            st.download_button("Download Percentile CSV", pct.to_csv(index=False).encode("utf-8"), "percentile_list__core025_northern_lights_v40.csv", "text/csv", use_container_width=True, on_click="ignore")
            st.download_button("Download Stream Method Tests CSV", st.session_state["stream_method_tests"].to_csv(index=False).encode("utf-8") if st.session_state["stream_method_tests"] is not None else b"", "stream_method_tests__core025_northern_lights_v40.csv", "text/csv", use_container_width=True, on_click="ignore")
        with c4:
            st.download_button("Download Stream Reduction CSV", stream_diag.to_csv(index=False).encode("utf-8"), "stream_reduction__core025_northern_lights_v40.csv", "text/csv", use_container_width=True, on_click="ignore")
            st.download_button("Download Single Row Perf CSV", st.session_state["row_single_perf"].to_csv(index=False).encode("utf-8") if st.session_state["row_single_perf"] is not None else b"", "single_row_performance__core025_northern_lights_v40.csv", "text/csv", use_container_width=True, on_click="ignore")


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

    st.markdown("### Stream / Row Test Downloads")
    if st.session_state["stream_method_tests"] is not None:
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button(
                "Download Stream Method Tests CSV",
                st.session_state["stream_method_tests"].to_csv(index=False).encode("utf-8"),
                "stream_method_tests__core025_northern_lights_v40.csv",
                "text/csv",
                use_container_width=True,
                on_click="ignore",
            )
            st.download_button(
                "Download Stream Method Tests TXT",
                st.session_state["stream_method_tests"].to_csv(index=False).encode("utf-8"),
                "stream_method_tests__core025_northern_lights_v40.txt",
                "text/plain",
                use_container_width=True,
                on_click="ignore",
            )
        with d2:
            st.download_button(
                "Download Single Row Performance CSV",
                st.session_state["row_single_perf"].to_csv(index=False).encode("utf-8") if st.session_state["row_single_perf"] is not None else b"",
                "single_row_performance__core025_northern_lights_v40.csv",
                "text/csv",
                use_container_width=True,
                on_click="ignore",
            )
            st.download_button(
                "Download Single Row Performance TXT",
                st.session_state["row_single_perf"].to_csv(index=False).encode("utf-8") if st.session_state["row_single_perf"] is not None else b"",
                "single_row_performance__core025_northern_lights_v40.txt",
                "text/plain",
                use_container_width=True,
                on_click="ignore",
            )
        with d3:
            st.download_button(
                "Download Percentile List CSV",
                st.session_state["percentile_df"].to_csv(index=False).encode("utf-8") if st.session_state["percentile_df"] is not None else b"",
                "percentile_list__core025_northern_lights_v40.csv",
                "text/csv",
                use_container_width=True,
                on_click="ignore",
            )
            st.download_button(
                "Download Percentile List TXT",
                st.session_state["percentile_df"].to_csv(index=False).encode("utf-8") if st.session_state["percentile_df"] is not None else b"",
                "percentile_list__core025_northern_lights_v40.txt",
                "text/plain",
                use_container_width=True,
                on_click="ignore",
            )
        with d4:
            st.download_button(
                "Download Stream Reduction CSV",
                stream_diag.to_csv(index=False).encode("utf-8"),
                "stream_reduction__core025_northern_lights_v40.csv",
                "text/csv",
                use_container_width=True,
                on_click="ignore",
            )
            st.download_button(
                "Download Stream Reduction TXT",
                stream_diag.to_csv(index=False).encode("utf-8"),
                "stream_reduction__core025_northern_lights_v40.txt",
                "text/plain",
                use_container_width=True,
                on_click="ignore",
            )
    else:
        st.info("Run the Walk-forward Lab first to populate historical stream-method and single-row performance tests.")


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

### v40 decision-layer correction
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

### v40 stream / row scoring + rare Top3 rescue additions
- Adds PlayType and RecommendedPlay display logic.
- Full capture display can bold Top1+Top2, while risk-gate mode bolds Top2 only when TOP2_REQUIRED.
- Adds row-by-row percentile scoring: SingleRow, RowPercentile, RowBucket10, StreamTier.
- Adds a Stream / Row Tests tab for Top-N, percentile bands, tiers, and single-row historical performance.
- Adds Rare Top3 Rescue for miss/outlier rows before any Top2 waste trimming.
- Adds UniversalFamilyScore, scaled for future cross-family app comparison.
- Adds explicit Stream / Row Test download buttons for CSV and TXT exports.

### Required files
For daily prediction:
1. Full history
2. Promoted separator library
3. FULL TRUTH file recommended for intelligence layer
4. Last 24h optional

For lab:
Same files, then run Walk-forward Lab.
""")
