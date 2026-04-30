
# BUILD: core025_v54_structured_straight_predictor__2026-04-30_FULL

from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


BUILD_LABEL = "core025_v54_structured_straight_predictor__2026-04-30_FULL"
CORE_MEMBERS = {"0025", "0225", "0255"}


# =========================
# BASIC HELPERS
# =========================

def norm4(x) -> Optional[str]:
    digs = re.findall(r"\d", str(x))
    if len(digs) < 4:
        return None
    return "".join(digs[:4]).zfill(4)


def box4(x) -> Optional[str]:
    n = norm4(x)
    return "".join(sorted(n)) if n else None


def digits4(x) -> List[int]:
    return [int(d) for d in str(x).zfill(4)]


def digit_sum(x) -> int:
    return sum(digits4(x))


def root_sum(x) -> int:
    s = digit_sum(x)
    return s % 9 if s % 9 else 9


def spread(x) -> int:
    d = digits4(x)
    return max(d) - min(d)


def bucket(value: int, cuts: List[int]) -> str:
    prev = None
    for c in cuts:
        if value <= c:
            return f"<= {c}" if prev is None else f"{prev+1}-{c}"
        prev = c
    return f"> {cuts[-1]}"


def parity_pattern(x) -> str:
    return "".join("E" if d % 2 == 0 else "O" for d in digits4(x))


def highlow_pattern(x) -> str:
    return "".join("H" if d >= 5 else "L" for d in digits4(x))


def repeat_shape(x) -> str:
    counts = sorted(Counter(str(x).zfill(4)).values(), reverse=True)
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


def mirror_traits(x) -> Dict[str, str]:
    ds = set(digits4(x))
    present = []
    for a, b in [(0,5), (1,6), (2,7), (3,8), (4,9)]:
        if a in ds and b in ds:
            present.append(f"{a}{b}")
    return {
        "mirror_count": str(len(present)),
        "has_mirror": str(int(bool(present))),
        "mirror_pairs": "|".join(present) if present else "none",
    }


def double_positions(perm: str) -> str:
    c = Counter(str(perm).zfill(4))
    doubles = [d for d, n in c.items() if n == 2]
    if not doubles:
        return "none"
    d = doubles[0]
    return "-".join(str(i+1) for i, x in enumerate(str(perm).zfill(4)) if x == d)


def member_of_perm(perm: str) -> Optional[str]:
    b = box4(perm)
    return b if b in CORE_MEMBERS else None


def double_digit_for_member(member: str) -> str:
    return {"0025": "0", "0225": "2", "0255": "5"}.get(member, "")


def all_perms_for_member(member: str) -> List[str]:
    from itertools import permutations
    return sorted({"".join(p) for p in permutations(member, 4)})


def build_seed_trait_values(seed: str) -> Dict[str, str]:
    seed = str(seed).zfill(4)
    d = digits4(seed)
    mt = mirror_traits(seed)
    vals = {
        "seed_sum": str(digit_sum(seed)),
        "SeedSum": str(digit_sum(seed)),
        "seed_sum_bucket": bucket(digit_sum(seed), [6, 10, 14, 18, 22, 26, 30]),
        "seed_root": str(root_sum(seed)),
        "seed_spread": str(spread(seed)),
        "seed_spread_bucket": bucket(spread(seed), [2, 4, 6, 8]),
        "seed_parity": parity_pattern(seed),
        "SeedParity": parity_pattern(seed),
        "seed_highlow": highlow_pattern(seed),
        "SeedHighLow": highlow_pattern(seed),
        "seed_repeat_shape": repeat_shape(seed),
        "SeedRepeatShape": repeat_shape(seed),
        "seed_even_count": str(sum(1 for x in d if x % 2 == 0)),
        "seed_high_count": str(sum(1 for x in d if x >= 5)),
        "core_digit_count": str(sum(1 for ch in seed if ch in "025")),
        "has0": str(int("0" in seed)),
        "has2": str(int("2" in seed)),
        "has5": str(int("5" in seed)),
        "has9": str(int("9" in seed)),
        "first_digit": seed[0],
        "last_digit": seed[-1],
        "first_pair": seed[:2],
        "last_pair": seed[2:],
        "mirror_count": mt["mirror_count"],
        "has_mirror": mt["has_mirror"],
        "mirror_pairs": mt["mirror_pairs"],
    }
    for i, ch in enumerate(seed):
        vals[f"S{i+1}"] = ch
        vals[f"S{i+1}_is_core"] = str(int(ch in "025"))
    return vals


def build_trait_tokens(seed: str) -> set:
    s = str(seed).zfill(4)
    vals = build_seed_trait_values(s)
    tokens = [f"{k}={v}" for k, v in vals.items()]

    for i in range(3):
        p = s[i:i+2]
        tokens.append(f"adj_pair={p}")
        tokens.append(f"touching_unordered_pair={''.join(sorted(p))}")

    for i in range(4):
        for j in range(i + 1, 4):
            tokens.append(f"ordered_pair_any={s[i] + s[j]}")
            tokens.append(f"unordered_pair_any={''.join(sorted(s[i] + s[j]))}")
            tokens.append(f"S{i+1}{j+1}={s[i] + s[j]}")
            tokens.append(f"S{i+1}{j+1}_unordered={''.join(sorted(s[i] + s[j]))}")

    for comb in [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]:
        tokens.append(f"unordered_triplet_any={''.join(sorted(''.join(s[i] for i in comb)))}")

    return set(tokens)


def stack_matches(tokens: set, stack: str) -> bool:
    parts = [p.strip() for p in str(stack).split("&&")]
    return bool(parts) and all(p in tokens for p in parts)


# =========================
# LOADERS / NORMALIZERS
# =========================

def load_table(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    raw = uploaded.getvalue()
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        try:
            return pd.read_csv(io.BytesIO(raw), sep="\t")
        except Exception:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")


def normalize_straight_export(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "StraightPermutation" not in df.columns:
        for c in df.columns:
            cl = c.lower()
            if "straight" in cl and ("perm" in cl or "play" in cl):
                df = df.rename(columns={c: "StraightPermutation"})
                break
    if "seed" not in df.columns:
        for c in df.columns:
            if c.lower() in ["seed", "lastresult", "last_result", "last result"]:
                df = df.rename(columns={c: "seed"})
                break
    if "PlaylistRank" not in df.columns:
        if "StreamRank" in df.columns:
            df["PlaylistRank"] = df["StreamRank"]
        else:
            df["PlaylistRank"] = 999
    if "StreamStraightRank" not in df.columns:
        df["StreamStraightRank"] = df.groupby("PlaylistRank").cumcount() + 1
    if "StraightConfidenceScore" not in df.columns:
        df["StraightConfidenceScore"] = 0.0
    if "StreamKey" not in df.columns:
        df["StreamKey"] = "Unknown"

    required = ["StraightPermutation", "seed", "PlaylistRank", "StreamKey"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Daily straight export missing required columns: {missing}")

    df["StraightPermutation"] = df["StraightPermutation"].map(norm4)
    df["seed"] = df["seed"].map(norm4)
    df = df.dropna(subset=["StraightPermutation", "seed"]).copy()
    df["PlaylistRank"] = pd.to_numeric(df["PlaylistRank"], errors="coerce").fillna(999).astype(int)
    df["StreamStraightRank"] = pd.to_numeric(df["StreamStraightRank"], errors="coerce").fillna(999).astype(int)
    df["StraightConfidenceScore"] = pd.to_numeric(df["StraightConfidenceScore"], errors="coerce").fillna(0.0)
    df["PredictedMemberFromPerm"] = df["StraightPermutation"].map(member_of_perm)
    df["DoublePositions"] = df["StraightPermutation"].map(double_positions)
    return df


def normalize_rules(df: pd.DataFrame, min_hits: int, min_conf: float, min_lift: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    required = {"TargetPermutation", "TraitStack", "Hits", "TraitTotal", "ConfidencePct", "Lift", "RuleScore"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stacked rules file missing columns: {sorted(missing)}")
    out = df.copy()
    out["TargetPermutation"] = out["TargetPermutation"].map(norm4)
    for c in ["Hits", "TraitTotal", "ConfidencePct", "Lift", "RuleScore"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out = out[
        out["TargetPermutation"].notna()
        & out["Hits"].ge(min_hits)
        & out["ConfidencePct"].ge(min_conf)
        & out["Lift"].ge(min_lift)
    ].copy()
    return out.sort_values(["RuleScore", "Hits", "Lift"], ascending=False).reset_index(drop=True)


def normalize_seed_trait_to_member(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "WinMember" not in out.columns:
        for c in out.columns:
            if "member" in c.lower():
                out = out.rename(columns={c: "WinMember"})
                break
    if "RatePct" not in out.columns:
        if "ConfidencePct" in out.columns:
            out["RatePct"] = out["ConfidencePct"]
        elif {"Hits", "Total"} <= set(out.columns):
            out["RatePct"] = pd.to_numeric(out["Hits"], errors="coerce") / pd.to_numeric(out["Total"], errors="coerce") * 100
    return out


def normalize_double_position(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    # expected: WinMember, DoubleDigit, DoublePositions, Hits, Total, RatePct
    for c in ["Hits", "Total", "RatePct"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out


def normalize_position_map(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    # expected: SeedPos, SeedDigit, WinnerPos, WinnerDigit, Hits/Count/RatePct
    if "Hits" not in out.columns and "Count" in out.columns:
        out = out.rename(columns={"Count": "Hits"})
    for c in ["Hits", "Total", "RatePct"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out


# =========================
# STRUCTURED PREDICTOR SCORING
# =========================

def score_member_layer(seed: str, perm: str, seed_member_df: pd.DataFrame, base_member_weight: float) -> Tuple[float, str]:
    member = member_of_perm(perm)
    if not member:
        return 0.0, ""
    if seed_member_df.empty:
        return 0.0, ""

    traits = build_seed_trait_values(seed)
    boost = 0.0
    matches = []

    # Handles both seed_trait_to_member and seed_position_digit_to_member style tables.
    for _, r in seed_member_df.iterrows():
        wm = str(r.get("WinMember", ""))
        if wm != member:
            continue

        rate = float(r.get("RatePct", 0) or 0)
        total = float(r.get("Total", r.get("TraitTotal", 0)) or 0)
        if rate <= 0:
            continue

        matched = False
        if "Trait" in r and "TraitValue" in r:
            trait = str(r["Trait"])
            val = str(r["TraitValue"])
            if traits.get(trait) == val:
                matched = True
        elif "SeedPos" in r and "SeedDigit" in r:
            pos = str(r["SeedPos"])
            sd = str(r["SeedDigit"])
            if traits.get(pos) == sd:
                matched = True
        elif "SeedSum" in r:
            if traits.get("SeedSum") == str(r["SeedSum"]):
                matched = True
        elif "SeedParity" in r:
            if traits.get("SeedParity") == str(r["SeedParity"]):
                matched = True
        elif "SeedHighLow" in r:
            if traits.get("SeedHighLow") == str(r["SeedHighLow"]):
                matched = True

        if matched:
            # total tempers tiny-sample rules
            sample_factor = min(total / 20.0, 1.0) if total else 0.25
            add = (rate / 100.0) * base_member_weight * sample_factor
            boost += add
            matches.append(f'{wm}:{rate:.1f}% n={int(total)}')

    return round(boost, 4), " | ".join(matches[:4])


def score_double_position_layer(seed: str, perm: str, double_pos_df: pd.DataFrame, seed_to_double_df: pd.DataFrame, weight: float) -> Tuple[float, str]:
    member = member_of_perm(perm)
    dp = double_positions(perm)
    dd = double_digit_for_member(member or "")
    boost = 0.0
    matches = []

    if not double_pos_df.empty:
        sub = double_pos_df[
            (double_pos_df.get("WinMember", "").astype(str) == str(member))
            & (double_pos_df.get("DoublePositions", "").astype(str) == str(dp))
        ] if {"WinMember", "DoublePositions"} <= set(double_pos_df.columns) else pd.DataFrame()
        for _, r in sub.iterrows():
            rate = float(r.get("RatePct", 0) or 0)
            total = float(r.get("Total", r.get("Hits", 0)) or 0)
            sample_factor = min(total / 20.0, 1.0) if total else 0.25
            add = (rate / 100.0) * weight * sample_factor
            boost += add
            matches.append(f"global_double_pos={dp}:{rate:.1f}% n={int(total)}")

    if not seed_to_double_df.empty:
        traits = build_seed_trait_values(seed)
        for _, r in seed_to_double_df.iterrows():
            if str(r.get("WinMember", "")) != str(member):
                continue
            if str(r.get("DoublePositions", "")) != str(dp):
                continue
            trait = str(r.get("Trait", ""))
            val = str(r.get("TraitValue", ""))
            if trait and val and traits.get(trait) == val:
                rate = float(r.get("RatePct", 0) or 0)
                total = float(r.get("Total", r.get("Hits", 0)) or 0)
                sample_factor = min(total / 20.0, 1.0) if total else 0.25
                add = (rate / 100.0) * weight * sample_factor
                boost += add
                matches.append(f"{trait}={val}->{dp}:{rate:.1f}% n={int(total)}")

    return round(boost, 4), " | ".join(matches[:4])


def score_position_map_layer(seed: str, perm: str, pos_map_df: pd.DataFrame, weight: float) -> Tuple[float, str]:
    if pos_map_df.empty:
        return 0.0, ""
    seed = str(seed).zfill(4)
    perm = str(perm).zfill(4)
    boost = 0.0
    matches = []

    # Use same-position and general Sx->Wy files if provided.
    for si in range(4):
        seed_pos = f"S{si+1}"
        seed_digit = seed[si]
        for wi in range(4):
            winner_pos = f"W{wi+1}"
            winner_digit = perm[wi]
            sub = pd.DataFrame()

            if {"SeedPos", "SeedDigit", "WinnerPos", "WinnerDigit"} <= set(pos_map_df.columns):
                sub = pos_map_df[
                    (pos_map_df["SeedPos"].astype(str) == seed_pos)
                    & (pos_map_df["SeedDigit"].astype(str) == seed_digit)
                    & (pos_map_df["WinnerPos"].astype(str) == winner_pos)
                    & (pos_map_df["WinnerDigit"].astype(str) == winner_digit)
                ]
            elif {"Position", "SeedDigitAtPosition", "WinnerDigitAtSamePosition"} <= set(pos_map_df.columns) and si == wi:
                sub = pos_map_df[
                    (pos_map_df["Position"].astype(str) == f"P{si+1}")
                    & (pos_map_df["SeedDigitAtPosition"].astype(str) == seed_digit)
                    & (pos_map_df["WinnerDigitAtSamePosition"].astype(str) == winner_digit)
                ]

            for _, r in sub.iterrows():
                rate = float(r.get("RatePct", 0) or 0)
                total = float(r.get("Total", r.get("Hits", 0)) or 0)
                if rate <= 0:
                    continue
                sample_factor = min(total / 25.0, 1.0) if total else 0.25
                add = (rate / 100.0) * weight * sample_factor / 4.0
                boost += add
                matches.append(f"{seed_pos}{seed_digit}->{winner_pos}{winner_digit}:{rate:.1f}%")

    return round(boost, 4), " | ".join(matches[:6])


def score_stacked_rule_layer(seed: str, perm: str, rules_df: pd.DataFrame, weight: float, max_boost: float) -> Tuple[float, int, str]:
    if rules_df.empty:
        return 0.0, 0, ""
    perm = norm4(perm)
    tokens = build_trait_tokens(seed)
    sub = rules_df[rules_df["TargetPermutation"].astype(str) == str(perm)].head(400)
    total = 0.0
    matches = []
    for _, r in sub.iterrows():
        if stack_matches(tokens, r["TraitStack"]):
            raw = (
                float(r["RuleScore"]) * 0.18
                + float(r["Hits"]) * 0.65
                + float(r["Lift"]) * 0.20
                + float(r["ConfidencePct"]) * 0.035
            )
            total += raw
            matches.append(f'{r["TraitStack"]} [h={int(r["Hits"])}, conf={r["ConfidencePct"]:.1f}, lift={r["Lift"]:.2f}]')
    total = min(total * weight, max_boost)
    return round(total, 4), len(matches), " | ".join(matches[:5])


def predict_structured(
    straight_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    seed_member_df: pd.DataFrame,
    seed_pos_member_df: pd.DataFrame,
    double_pos_df: pd.DataFrame,
    seed_to_double_df: pd.DataFrame,
    pos_map_df: pd.DataFrame,
    same_pos_df: pd.DataFrame,
    stacked_weight: float,
    member_weight: float,
    double_weight: float,
    pos_weight: float,
    max_stack_boost: float,
) -> pd.DataFrame:
    out = straight_df.copy()
    # Merge member trait files for a broader member layer.
    member_tables = []
    if not seed_member_df.empty:
        member_tables.append(seed_member_df)
    if not seed_pos_member_df.empty:
        member_tables.append(seed_pos_member_df)
    member_all = pd.concat(member_tables, ignore_index=True) if member_tables else pd.DataFrame()

    pos_tables = []
    if not pos_map_df.empty:
        pos_tables.append(pos_map_df)
    if not same_pos_df.empty:
        pos_tables.append(same_pos_df)
    pos_all = pd.concat(pos_tables, ignore_index=True) if pos_tables else pd.DataFrame()

    rows = []
    for _, row in out.iterrows():
        seed = str(row["seed"]).zfill(4)
        perm = str(row["StraightPermutation"]).zfill(4)
        base = float(row.get("StraightConfidenceScore", 0) or 0)

        member_boost, member_matches = score_member_layer(seed, perm, member_all, member_weight)
        double_boost, double_matches = score_double_position_layer(seed, perm, double_pos_df, seed_to_double_df, double_weight)
        pos_boost, pos_matches = score_position_map_layer(seed, perm, pos_all, pos_weight)
        stack_boost, stack_hits, stack_matches = score_stacked_rule_layer(seed, perm, rules_df, stacked_weight, max_stack_boost)

        total = base + member_boost + double_boost + pos_boost + stack_boost

        rr = row.to_dict()
        rr.update({
            "v54_BaseStraightScore": round(base, 4),
            "v54_MemberBoost": member_boost,
            "v54_MemberMatches": member_matches,
            "v54_DoublePositionBoost": double_boost,
            "v54_DoublePositionMatches": double_matches,
            "v54_PositionMapBoost": pos_boost,
            "v54_PositionMapMatches": pos_matches,
            "v54_StackedRuleBoost": stack_boost,
            "v54_StackedRuleHits": stack_hits,
            "v54_StackedRuleMatches": stack_matches,
            "v54_TotalPredictorScore": round(total, 4),
        })
        rows.append(rr)

    scored = pd.DataFrame(rows)
    scored = scored.sort_values(
        ["PlaylistRank", "v54_TotalPredictorScore", "v54_StackedRuleHits", "StraightConfidenceScore", "StreamStraightRank"],
        ascending=[True, False, False, False, True]
    ).copy()
    scored["v54_StreamStraightRank"] = scored.groupby("PlaylistRank").cumcount() + 1
    return scored


def build_sections(scored: pd.DataFrame, stream_depth: int, best_depth: int, optional_depth: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = scored[scored["PlaylistRank"].le(stream_depth)].copy()

    box = eligible.sort_values("PlaylistRank").drop_duplicates("PlaylistRank").copy()
    box["PlaylistSection"] = "BOX_PLAYS"
    box_cols = [c for c in [
        "PlaylistSection", "PlaylistRank", "StreamKey", "State", "Game", "seed",
        "BoxPlayType", "BoxRecommendedPlay", "PredictedMember", "Top2_pred", "ThirdMember",
        "RowPercentile", "SingleRow", "StreamTier"
    ] if c in box.columns]

    best = eligible[eligible["v54_StreamStraightRank"].le(best_depth)].copy()
    best["PlaylistSection"] = "BEST_PRACTICE_V54_STRAIGHTS"

    optional = eligible[
        eligible["v54_StreamStraightRank"].gt(best_depth)
        & eligible["v54_StreamStraightRank"].le(optional_depth)
    ].copy()
    optional["PlaylistSection"] = "OPTIONAL_EXPANDED_V54_STRAIGHTS"

    straight_cols = [c for c in [
        "PlaylistSection", "PlaylistRank", "StreamKey", "seed", "BoxRecommendedPlay",
        "StraightPermutation", "v54_StreamStraightRank", "v54_TotalPredictorScore",
        "v54_BaseStraightScore", "v54_MemberBoost", "v54_DoublePositionBoost",
        "v54_PositionMapBoost", "v54_StackedRuleBoost", "v54_StackedRuleHits",
        "v54_MemberMatches", "v54_DoublePositionMatches", "v54_PositionMapMatches",
        "v54_StackedRuleMatches", "StreamStraightRank", "StraightConfidenceScore",
        "RowPercentile", "SingleRow"
    ] if c in scored.columns or c == "PlaylistSection"]

    return box[box_cols], best[straight_cols], optional[straight_cols]


def make_zip(box, best, optional, scored, loaded_manifest):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("v54_REPORT_MANIFEST.csv", loaded_manifest.to_csv(index=False))
        zf.writestr("v54_box_playlist.csv", box.to_csv(index=False))
        zf.writestr("v54_best_practice_structured_straights.csv", best.to_csv(index=False))
        zf.writestr("v54_optional_expanded_structured_straights.csv", optional.to_csv(index=False))
        zf.writestr("v54_all_structured_scored_straights.csv", scored.to_csv(index=False))
    bio.seek(0)
    return bio.getvalue()


# =========================
# UI
# =========================

st.set_page_config(page_title="Core025 v54 Structured Straight Predictor", layout="wide")
st.title("Core025 v54 Structured Straight Predictor")
st.caption(f"BUILD: {BUILD_LABEL}")

st.info(
    "This is the structured straight layer: member prediction + double-position prediction + positional mapping + stacked permutation rules. "
    "It uses your mined reports and today’s straight-ranked export."
)

with st.sidebar:
    st.header("Required Inputs")
    straight_file = st.file_uploader("Daily straight-ranked export", type=["csv", "txt", "tsv"])
    stacked_rules_file = st.file_uploader("Stacked permutation separator rules", type=["csv", "txt", "tsv"])

    st.header("Strongly Recommended Inputs")
    seed_trait_member_file = st.file_uploader("seed_trait_to_member.csv", type=["csv", "txt", "tsv"])
    seed_position_member_file = st.file_uploader("seed_position_digit_to_member.csv", type=["csv", "txt", "tsv"])
    double_position_file = st.file_uploader("double_position_summary.csv", type=["csv", "txt", "tsv"])
    seed_trait_double_file = st.file_uploader("seed_trait_to_double_position.csv", type=["csv", "txt", "tsv"])
    position_map_file = st.file_uploader("seed_position_to_winner_position_digit.csv", type=["csv", "txt", "tsv"])
    same_position_file = st.file_uploader("same_position_seed_digit_to_winner_digit.csv", type=["csv", "txt", "tsv"])

    st.header("Rule Filters")
    min_hits = st.slider("Stacked min hits", 1, 10, 2)
    min_conf = st.slider("Stacked min confidence %", 0, 100, 30, step=5)
    min_lift = st.slider("Stacked min lift", 0.0, 10.0, 1.5, step=0.5)

    st.header("Layer Weights")
    member_weight = st.slider("Member layer weight", 0.0, 100.0, 20.0, step=1.0)
    double_weight = st.slider("Double-position layer weight", 0.0, 100.0, 30.0, step=1.0)
    pos_weight = st.slider("Position-map layer weight", 0.0, 100.0, 25.0, step=1.0)
    stacked_weight = st.slider("Stacked-rule layer weight", 0.0, 3.0, 1.0, step=0.1)
    max_stack_boost = st.slider("Max stacked-rule boost", 0.0, 100.0, 60.0, step=5.0)

    st.header("Playlist")
    stream_depth = st.slider("Stream depth", 1, 30, 15)
    best_depth = st.slider("Best-practice straights per stream", 1, 20, 6)
    optional_depth = st.slider("Optional depth per stream", 1, 24, 15)

if straight_file is None or stacked_rules_file is None:
    st.warning("Upload at least the daily straight-ranked export and stacked permutation separator rules.")
    st.stop()

try:
    straight_df = normalize_straight_export(load_table(straight_file))
    rules_df = normalize_rules(load_table(stacked_rules_file), min_hits=min_hits, min_conf=float(min_conf), min_lift=float(min_lift))

    seed_trait_member_df = normalize_seed_trait_to_member(load_table(seed_trait_member_file))
    seed_position_member_df = normalize_seed_trait_to_member(load_table(seed_position_member_file))
    double_position_df = normalize_double_position(load_table(double_position_file))
    seed_trait_double_df = normalize_double_position(load_table(seed_trait_double_file))
    position_map_df = normalize_position_map(load_table(position_map_file))
    same_position_df = normalize_position_map(load_table(same_position_file))
except Exception as e:
    st.error(f"Load/normalize failed: {e}")
    st.stop()

if rules_df.empty:
    st.error("No stacked rules passed the selected filters.")
    st.stop()

manifest = pd.DataFrame([
    {"Input": "daily_straight_ranked_export", "LoadedRows": len(straight_df), "Required": True},
    {"Input": "stacked_rules", "LoadedRows": len(rules_df), "Required": True},
    {"Input": "seed_trait_to_member", "LoadedRows": len(seed_trait_member_df), "Required": False},
    {"Input": "seed_position_digit_to_member", "LoadedRows": len(seed_position_member_df), "Required": False},
    {"Input": "double_position_summary", "LoadedRows": len(double_position_df), "Required": False},
    {"Input": "seed_trait_to_double_position", "LoadedRows": len(seed_trait_double_df), "Required": False},
    {"Input": "seed_position_to_winner_position_digit", "LoadedRows": len(position_map_df), "Required": False},
    {"Input": "same_position_seed_digit_to_winner_digit", "LoadedRows": len(same_position_df), "Required": False},
])

st.success(f"Loaded {len(straight_df):,} straight rows and {len(rules_df):,} stacked rules.")

scored = predict_structured(
    straight_df=straight_df,
    rules_df=rules_df,
    seed_member_df=seed_trait_member_df,
    seed_pos_member_df=seed_position_member_df,
    double_pos_df=double_position_df,
    seed_to_double_df=seed_trait_double_df,
    pos_map_df=position_map_df,
    same_pos_df=same_position_df,
    stacked_weight=stacked_weight,
    member_weight=member_weight,
    double_weight=double_weight,
    pos_weight=pos_weight,
    max_stack_boost=max_stack_boost,
)

box, best, optional = build_sections(scored, stream_depth=stream_depth, best_depth=best_depth, optional_depth=max(optional_depth, best_depth))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Streams", box["PlaylistRank"].nunique() if not box.empty and "PlaylistRank" in box.columns else 0)
c2.metric("Best straights", len(best))
c3.metric("Optional straights", len(optional))
c4.metric("Rules used", len(rules_df))

st.markdown("## SECTION 1 — BOX PLAYS")
st.dataframe(box, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("## SECTION 2 — BEST PRACTICE STRUCTURED STRAIGHTS")
st.dataframe(best, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("## SECTION 3 — OPTIONAL / AGGRESSIVE EXPANDED STRAIGHTS")
st.dataframe(optional, use_container_width=True, hide_index=True)

with st.expander("Input manifest", expanded=False):
    st.dataframe(manifest, use_container_width=True, hide_index=True)

with st.expander("Full v54 structured scored straight list", expanded=False):
    st.dataframe(scored, use_container_width=True, hide_index=True)

st.download_button(
    "Download v54 structured predictor ZIP",
    make_zip(box, best, optional, scored, manifest),
    "core025_v54_structured_straight_predictor_reports.zip",
    "application/zip",
    use_container_width=True,
)
