#!/usr/bin/env python3
# BUILD: core025_northern_lights__2026-04-23_v35_merged_live_truth

from __future__ import annotations

import io
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_northern_lights__2026-04-24_v35_strict_score_ranking_margin_gating"
MEMBERS = ["0025", "0225", "0255"]

st.set_page_config(page_title="Core025 Northern Lights 025", layout="wide")
st.title("Core025 Northern Lights — 025 Live + Lab")
st.caption(BUILD_MARKER)
st.warning(
    "025-only merged build v35: strict score ranking with margin-gated overrides only, live full-history + optional last-24h prediction workflow, "
    "Northern-Lights-style stream reduction, and v22/v31 truth-backed member engine."
)

for k in [
    "daily_playlist", "lab_results", "lab_summary", "stream_diag",
    "truth_rules", "loaded_meta", "percentile_df"
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
    strict_margin_lock: float = 5.0,
) -> Dict:
    """
    v35 strict score ranking:
    - Top1 is ALWAYS the highest score.
    - Top2 is ALWAYS the second-highest score.
    - GATED_TOP3 can only fire when the score margin is below strict_margin_lock.
    This prevents hidden/implicit 0025 bias from overriding strong score separation.
    """
    boosts, fired = apply_rules(row, rules_df, trait_weight)
    sorted_scores = sorted(boosts.items(), key=lambda x: x[1], reverse=True)

    top = sorted_scores[0][0]
    second = sorted_scores[1][0]
    third = sorted_scores[2][0]
    score1 = float(sorted_scores[0][1])
    score2 = float(sorted_scores[1][1])
    margin = score1 - score2
    seed = str(row.get("seed", ""))

    gated_allowed = margin <= float(strict_margin_lock)

    if gated_allowed and margin < min_margin:
        gate = False
        if (
            ("0" in seed and "9" in seed)
            or len(set(seed)) <= 2
            or any((d * 2) in seed for d in "0123456789")
            or "88" in seed or "99" in seed or "00" in seed
        ):
            gate = True
        if top == "0225" and second == "0255" and ("0" in seed or "9" in seed):
            gate = True
        if top == "0025" and second == "0255" and len(set(seed)) <= 3:
            gate = True
        if gate:
            top = third
            # After a top3 gate, preserve the strongest original score as Top2 safety unless duplicated.
            if second == top:
                second = sorted_scores[0][0]
            fired.append("GATED_TOP3 (margin-allowed)")

    return {
        "PredictedMember": top,
        "Top2_pred": second,
        "Margin": round(margin, 3),
        "ScoreRankLocked": int(not gated_allowed),
        "Fired_Rules": " | ".join(fired[:20]),
        "GATED_TOP3": int("GATED_TOP3 (margin-allowed)" in fired),
        "score_0025": boosts["0025"],
        "score_0225": boosts["0225"],
        "score_0255": boosts["0255"],
    }
