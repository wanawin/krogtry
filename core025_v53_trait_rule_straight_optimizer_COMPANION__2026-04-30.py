
# BUILD: core025_v53_trait_rule_straight_optimizer_COMPANION__2026-04-30

from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from typing import Optional, List

import pandas as pd
import streamlit as st

BUILD_LABEL = "core025_v53_trait_rule_straight_optimizer_COMPANION__2026-04-30"

st.set_page_config(page_title="Core025 v53 Trait Straight Optimizer", layout="wide")
st.title("Core025 v53 Trait-Rule Straight Optimizer")
st.caption(f"BUILD: {BUILD_LABEL}")

st.info(
    "Upload your daily straight-ranked export plus the v2.1 stacked separator rules. "
    "This companion app preserves your box/member output and re-ranks straight permutations using seed-trait separator rules."
)

# =========================
# TRAIT HELPERS
# =========================

def norm_perm_text(x) -> Optional[str]:
    digs = re.findall(r"\d", str(x))
    if len(digs) < 4:
        return None
    return "".join(digs[:4]).zfill(4)

def digits(seed: str) -> List[int]:
    return [int(x) for x in str(seed).zfill(4)]

def digit_sum(seed: str) -> int:
    return sum(digits(seed))

def root_sum(seed: str) -> int:
    sm = digit_sum(seed)
    return sm % 9 if sm % 9 != 0 else 9

def spread(seed: str) -> int:
    d = digits(seed)
    return max(d) - min(d)

def bucket(value: int, cuts: List[int]) -> str:
    prev = None
    for c in cuts:
        if value <= c:
            return f"<= {c}" if prev is None else f"{prev+1}-{c}"
        prev = c
    return f"> {cuts[-1]}"

def parity(seed: str) -> str:
    return "".join("E" if d % 2 == 0 else "O" for d in digits(seed))

def highlow(seed: str) -> str:
    return "".join("H" if d >= 5 else "L" for d in digits(seed))

def repeat_shape(seed: str) -> str:
    vals = sorted(Counter(str(seed).zfill(4)).values(), reverse=True)
    if vals == [1,1,1,1]:
        return "all_unique"
    if vals == [2,1,1]:
        return "one_pair"
    if vals == [2,2]:
        return "two_pair"
    if vals == [3,1]:
        return "triple"
    if vals == [4]:
        return "quad"
    return "other"

def mirror_traits(seed: str):
    ds = set(digits(seed))
    present = []
    for a, b in [(0,5), (1,6), (2,7), (3,8), (4,9)]:
        if a in ds and b in ds:
            present.append(f"{a}{b}")
    return {
        "mirror_count": str(len(present)),
        "has_mirror": str(int(bool(present))),
        "mirror_pairs": "|".join(present) if present else "none",
    }

def build_trait_tokens(seed: str):
    s = str(seed).zfill(4)
    d = digits(s)
    mt = mirror_traits(s)

    single = {
        "seed_sum": str(digit_sum(s)),
        "seed_sum_bucket": bucket(digit_sum(s), [6, 10, 14, 18, 22, 26, 30]),
        "seed_root": str(root_sum(s)),
        "seed_spread": str(spread(s)),
        "seed_spread_bucket": bucket(spread(s), [2, 4, 6, 8]),
        "seed_parity": parity(s),
        "seed_highlow": highlow(s),
        "seed_repeat_shape": repeat_shape(s),
        "seed_even_count": str(sum(1 for x in d if x % 2 == 0)),
        "seed_high_count": str(sum(1 for x in d if x >= 5)),
        "core_digit_count": str(sum(1 for ch in s if ch in "025")),
        "has0": str(int("0" in s)),
        "has2": str(int("2" in s)),
        "has5": str(int("5" in s)),
        "has9": str(int("9" in s)),
        "first_digit": s[0],
        "last_digit": s[-1],
        "first_pair": s[:2],
        "last_pair": s[2:],
        "mirror_count": mt["mirror_count"],
        "has_mirror": mt["has_mirror"],
        "mirror_pairs": mt["mirror_pairs"],
    }

    for i, ch in enumerate(s):
        single[f"S{i+1}"] = ch
        single[f"S{i+1}_is_core"] = str(int(ch in "025"))

    tokens = [f"{k}={v}" for k, v in single.items()]

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

    return sorted(set(tokens))

# =========================
# LOADERS
# =========================

def load_csv_or_txt(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        try:
            return pd.read_csv(io.BytesIO(raw), sep="\t")
        except Exception:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")

def load_rules(uploaded, min_hits, min_conf, min_lift):
    if uploaded is None:
        return pd.DataFrame()
    rules = load_csv_or_txt(uploaded)
    required = {"TargetPermutation", "TraitStack", "Hits", "TraitTotal", "ConfidencePct", "Lift", "RuleScore"}
    missing = required - set(rules.columns)
    if missing:
        raise ValueError(f"Rules file missing columns: {sorted(missing)}")
    rules = rules.copy()
    rules["TargetPermutation"] = rules["TargetPermutation"].map(norm_perm_text)
    for c in ["Hits", "TraitTotal", "ConfidencePct", "Lift", "RuleScore"]:
        rules[c] = pd.to_numeric(rules[c], errors="coerce").fillna(0)
    return rules[
        rules["TargetPermutation"].notna()
        & rules["Hits"].ge(min_hits)
        & rules["ConfidencePct"].ge(min_conf)
        & rules["Lift"].ge(min_lift)
    ].sort_values(["RuleScore", "Hits", "Lift"], ascending=False).reset_index(drop=True)

def normalize_straight_export(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standard v50/v51 columns expected, but normalize common variants.
    if "StraightPermutation" not in df.columns:
        for c in df.columns:
            if "straight" in c.lower() and "perm" in c.lower():
                df = df.rename(columns={c: "StraightPermutation"})
                break
    if "seed" not in df.columns:
        for c in df.columns:
            if c.lower() in ["seed", "lastresult", "last_result"]:
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
    required = ["StraightPermutation", "seed", "PlaylistRank", "StreamKey"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Straight export missing required columns: {missing}")
    df["StraightPermutation"] = df["StraightPermutation"].map(norm_perm_text)
    df["seed"] = df["seed"].map(norm_perm_text)
    df = df.dropna(subset=["StraightPermutation", "seed"]).copy()
    df["PlaylistRank"] = pd.to_numeric(df["PlaylistRank"], errors="coerce").fillna(999).astype(int)
    df["StreamStraightRank"] = pd.to_numeric(df["StreamStraightRank"], errors="coerce").fillna(999).astype(int)
    df["StraightConfidenceScore"] = pd.to_numeric(df["StraightConfidenceScore"], errors="coerce").fillna(0.0)
    return df

# =========================
# SCORING
# =========================

def stack_matches(tokens: set, stack: str) -> bool:
    parts = [p.strip() for p in str(stack).split("&&")]
    return bool(parts) and all(p in tokens for p in parts)

def score_with_rules(straights: pd.DataFrame, rules: pd.DataFrame, rule_weight: float, max_boost: float) -> pd.DataFrame:
    out = straights.copy()
    if rules.empty:
        out["TraitRuleBoost"] = 0.0
        out["TraitRuleHits"] = 0
        out["TraitRuleTopMatches"] = ""
        out["TraitOptimizedStraightScore"] = out["StraightConfidenceScore"]
        out["TraitOptimizedStreamStraightRank"] = out["StreamStraightRank"]
        return out

    by_perm = {p: g for p, g in rules.groupby("TargetPermutation")}
    token_cache = {}

    boosts = []
    hit_counts = []
    top_matches = []

    for _, row in out.iterrows():
        seed = str(row["seed"]).zfill(4)
        perm = str(row["StraightPermutation"]).zfill(4)
        if seed not in token_cache:
            token_cache[seed] = set(build_trait_tokens(seed))
        tokens = token_cache[seed]

        total_boost = 0.0
        matches = []
        grp = by_perm.get(perm)
        if grp is not None:
            for _, rr in grp.head(350).iterrows():
                if stack_matches(tokens, rr["TraitStack"]):
                    raw = (
                        float(rr["RuleScore"]) * 0.18
                        + float(rr["Hits"]) * 0.65
                        + float(rr["Lift"]) * 0.20
                        + float(rr["ConfidencePct"]) * 0.035
                    )
                    total_boost += raw
                    matches.append(f'{rr["TraitStack"]} [h={int(rr["Hits"])}, conf={rr["ConfidencePct"]:.1f}, lift={rr["Lift"]:.2f}]')

        total_boost = min(total_boost * rule_weight, max_boost)
        boosts.append(round(total_boost, 4))
        hit_counts.append(len(matches))
        top_matches.append(" | ".join(matches[:5]))

    out["TraitRuleBoost"] = boosts
    out["TraitRuleHits"] = hit_counts
    out["TraitRuleTopMatches"] = top_matches
    out["TraitOptimizedStraightScore"] = (out["StraightConfidenceScore"] + out["TraitRuleBoost"]).round(4)

    out = out.sort_values(
        ["PlaylistRank", "TraitOptimizedStraightScore", "StraightConfidenceScore", "StreamStraightRank"],
        ascending=[True, False, False, True]
    ).copy()
    out["TraitOptimizedStreamStraightRank"] = out.groupby("PlaylistRank").cumcount() + 1
    return out

def build_playlists(scored, stream_depth, best_depth, optional_depth):
    eligible = scored[scored["PlaylistRank"].le(stream_depth)].copy()
    box = eligible.sort_values("PlaylistRank").drop_duplicates("PlaylistRank").copy()
    box["PlaylistSection"] = "BOX_PLAYS"
    box_cols = [c for c in [
        "PlaylistSection", "PlaylistRank", "StreamKey", "State", "Game", "seed",
        "BoxPlayType", "BoxRecommendedPlay", "PredictedMember", "Top2_pred", "ThirdMember",
        "RowPercentile", "SingleRow", "StreamTier"
    ] if c in box.columns]
    best = eligible[eligible["TraitOptimizedStreamStraightRank"].le(best_depth)].copy()
    best["PlaylistSection"] = "BEST_PRACTICE_TRAIT_STRAIGHTS"
    opt = eligible[
        eligible["TraitOptimizedStreamStraightRank"].gt(best_depth)
        & eligible["TraitOptimizedStreamStraightRank"].le(optional_depth)
    ].copy()
    opt["PlaylistSection"] = "OPTIONAL_EXPANDED_TRAIT_STRAIGHTS"

    straight_cols = [c for c in [
        "PlaylistSection", "PlaylistRank", "StreamKey", "seed", "BoxRecommendedPlay",
        "StraightPermutation", "TraitOptimizedStreamStraightRank", "TraitOptimizedStraightScore",
        "TraitRuleBoost", "TraitRuleHits", "TraitRuleTopMatches",
        "StraightConfidenceScore", "StreamStraightRank", "RowPercentile", "SingleRow"
    ] if c in scored.columns or c == "PlaylistSection"]
    return box[box_cols], best[straight_cols], opt[straight_cols]

def make_zip(box, best, opt, scored, rules):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("v53_box_playlist.csv", box.to_csv(index=False))
        zf.writestr("v53_best_practice_trait_straights.csv", best.to_csv(index=False))
        zf.writestr("v53_optional_expanded_trait_straights.csv", opt.to_csv(index=False))
        zf.writestr("v53_all_trait_scored_straights.csv", scored.to_csv(index=False))
        zf.writestr("v53_loaded_separator_rules.csv", rules.to_csv(index=False))
    bio.seek(0)
    return bio.getvalue()

# =========================
# UI
# =========================

with st.sidebar:
    st.header("Inputs")
    straight_file = st.file_uploader("Upload daily straight-ranked export (.csv/.txt)", type=["csv", "txt", "tsv"])
    rules_file = st.file_uploader("Upload stacked separator rules CSV", type=["csv", "txt", "tsv"])

    st.header("Rule thresholds")
    min_hits = st.slider("Min rule hits", 1, 10, 2)
    min_conf = st.slider("Min confidence %", 0, 100, 30, step=5)
    min_lift = st.slider("Min lift", 0.0, 10.0, 1.5, step=0.5)
    rule_weight = st.slider("Rule boost weight", 0.0, 3.0, 1.0, step=0.1)
    max_boost = st.slider("Max rule boost", 0.0, 100.0, 60.0, step=5.0)

    st.header("Playlist settings")
    stream_depth = st.slider("Stream depth", 1, 30, 15)
    best_depth = st.slider("Best-practice straights per stream", 1, 20, 6)
    optional_depth = st.slider("Optional straight depth per stream", 1, 24, 15)

if not straight_file or not rules_file:
    st.warning("Upload both the daily straight-ranked export and the stacked separator rules CSV.")
    st.stop()

try:
    straight_raw = load_csv_or_txt(straight_file)
    straight_df = normalize_straight_export(straight_raw)
    rules = load_rules(rules_file, min_hits=min_hits, min_conf=float(min_conf), min_lift=float(min_lift))
except Exception as e:
    st.error(f"Load failed: {e}")
    st.stop()

if rules.empty:
    st.error("No separator rules passed current thresholds.")
    st.stop()

st.success(f"Loaded {len(straight_df):,} straight rows and {len(rules):,} usable separator rules.")

scored = score_with_rules(straight_df, rules, rule_weight=rule_weight, max_boost=max_boost)
box, best, opt = build_playlists(scored, stream_depth=stream_depth, best_depth=best_depth, optional_depth=max(optional_depth, best_depth))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Streams in playlist", box["PlaylistRank"].nunique() if not box.empty and "PlaylistRank" in box.columns else 0)
c2.metric("Best-practice straights", len(best))
c3.metric("Optional straights", len(opt))
c4.metric("Rules loaded", len(rules))

st.markdown("## SECTION 1 — BOX PLAYS")
st.dataframe(box, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("## SECTION 2 — BEST PRACTICE TRAIT-OPTIMIZED STRAIGHTS")
st.dataframe(best, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("## SECTION 3 — OPTIONAL / AGGRESSIVE EXPANDED STRAIGHTS")
st.dataframe(opt, use_container_width=True, hide_index=True)

with st.expander("Full v53 trait-scored straight list", expanded=False):
    st.dataframe(scored, use_container_width=True, hide_index=True)

st.download_button(
    "Download v53 unified playlist ZIP",
    make_zip(box, best, opt, scored, rules),
    "core025_v53_trait_optimized_daily_playlist.zip",
    "application/zip",
    use_container_width=True,
)
