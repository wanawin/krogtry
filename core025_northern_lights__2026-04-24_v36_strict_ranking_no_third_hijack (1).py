#!/usr/bin/env python3
# BUILD: core025_northern_lights__2026-04-23_v33_merged_live_truth

from __future__ import annotations

import io
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_northern_lights__2026-04-24_v36_strict_ranking_no_third_hijack"
MEMBERS = ["0025", "0225", "0255"]

st.set_page_config(page_title="Core025 Northern Lights 025", layout="wide")
st.title("Core025 Northern Lights — 025 Live + Lab")
st.caption(BUILD_MARKER)
st.warning(
    "025-only merged build v36: v34 scoring preserved, strict score ranking, no third-choice hijack, live full-history + optional last-24h prediction workflow, "
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
    enable_margin_swap: bool = False,
) -> Dict:
    """
    v36 decision layer:
    - Preserve v34 scoring and rule firing.
    - Top1 is the highest score.
    - Top2 is the second-highest score.
    - No GATED_TOP3 third-choice hijack.
    - Optional cautious Top1/Top2 swap only when margin is tight AND seed is a danger seed.
    """
    boosts, fired = apply_rules(row, rules_df, trait_weight)
    sorted_scores = sorted(boosts.items(), key=lambda x: x[1], reverse=True)

    top = sorted_scores[0][0]
    second = sorted_scores[1][0]
    margin = float(sorted_scores[0][1]) - float(sorted_scores[1][1])
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

    return {
        "PredictedMember": top,
        "Top2_pred": second,
        "Margin": round(margin, 3),
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
    truth_min_rate = st.slider("Truth-mined min rate", 0.50, 0.95, 0.76, step=0.01)
    truth_min_support = st.slider("Truth-mined min support", 1, 25, 5)

tab_daily, tab_lab, tab_help = st.tabs(["Daily Prediction", "Walk-forward Lab", "Notes"])

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
            pick = v22_pick(row, rules, trait_weight, min_margin, enable_margin_swap) if not rules.empty else {
                "PredictedMember": "0025", "Top2_pred": "0225", "Margin": 0.0, "Fired_Rules": "",
                "GATED_TOP3": 0, "score_0025": 0, "score_0225": 0, "score_0255": 0
            }
            scored.append({**row.to_dict(), **pick})
        playlist = pd.DataFrame(scored)
        if not playlist.empty:
            playlist["Top1_BOLD"] = "**" + playlist["PredictedMember"].astype(str) + "**"
            playlist["RecommendedPlay"] = playlist["Top1_BOLD"]
            merge_cols = [c for c in ["StreamKey", "hit_density", "rank", "StreamRank", "kept"] if c in stream_diag.columns]
            playlist = playlist.merge(stream_diag[merge_cols], on="StreamKey", how="left")
            playlist = playlist.sort_values(["hit_density", "Margin"], ascending=[False, False]).reset_index(drop=True)
            playlist["PlaylistRank"] = playlist.index + 1
            playlist["margin_percentile"] = playlist["Margin"].rank(pct=True, method="average").round(4)
            front_cols = [
                "PlaylistRank", "StreamRank", "StreamKey", "State", "Game", "LastResult",
                "Top1_BOLD", "Top2_pred", "PredictedMember", "Margin", "DecisionMode",
                "score_0025", "score_0225", "score_0255", "hit_density", "Fired_Rules"
            ]
            playlist = playlist[[c for c in front_cols if c in playlist.columns] + [c for c in playlist.columns if c not in front_cols]]
        st.session_state["daily_playlist"] = playlist
        prog.progress(100, text="Done")
        status.update(label="Daily prediction complete", state="complete", expanded=False)

    if st.session_state["daily_playlist"] is not None:
        playlist = st.session_state["daily_playlist"]
        st.dataframe(playlist, use_container_width=True, hide_index=True)
        a, b = st.columns(2)
        with a:
            st.download_button("Download Daily Playlist CSV", playlist.to_csv(index=False).encode("utf-8"), "daily_playlist__core025_northern_lights_v36.csv", "text/csv", use_container_width=True, on_click="ignore")
        with b:
            st.download_button("Download Daily Playlist TXT", playlist.to_csv(index=False).encode("utf-8"), "daily_playlist__core025_northern_lights_v36.txt", "text/plain", use_container_width=True, on_click="ignore")

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
            pick = v22_pick(row, rules, trait_weight, min_margin, enable_margin_swap) if not rules.empty else {
                "PredictedMember": "0025", "Top2_pred": "0225", "Margin": 0.0, "Fired_Rules": "",
                "GATED_TOP3": 0, "score_0025": 0, "score_0225": 0, "score_0255": 0
            }
            top1 = int(pick["PredictedMember"] == row["TrueMember"])
            top2_needed = int(top1 == 0 and pick["Top2_pred"] == row["TrueMember"])
            miss = int(top1 == 0 and top2_needed == 0)
            scored.append({**row.to_dict(), **pick, "Top1_Correct": top1, "Needed_Top2": top2_needed, "Miss": miss})
        res = pd.DataFrame(scored)
        total = len(res)
        top1 = int(res["Top1_Correct"].sum()) if total else 0
        needed = int(res["Needed_Top2"].sum()) if total else 0
        miss = int(res["Miss"].sum()) if total else 0
        capture = (top1 + needed) / total * 100 if total else 0.0
        summary = pd.DataFrame([{
            "total_rows": total,
            "top1": top1,
            "top1_pct": round((top1 / total * 100), 2) if total else 0.0,
            "needed_top2": needed,
            "top2_burden_pct": round((needed / total * 100), 2) if total else 0.0,
            "miss": miss,
            "capture_pct": round(capture, 2),
            "gated_top3": int(res["GATED_TOP3"].sum()) if total else 0,
            "margin_swaps": int(res["MarginSwap"].sum()) if total and "MarginSwap" in res.columns else 0,
            "kept_streams": len(kept_streams),
            "active_rules": len(rules),
            "truth_mined_rules": len(new_truth_rules),
            "decision_layer": "strict_score_ranking_no_third_hijack",
            "margin_swap_enabled": enable_margin_swap,
        }])
        pct = res.copy()
        if not pct.empty:
            pct["margin_percentile"] = pct["Margin"].rank(pct=True, method="average").round(4)
        st.session_state["lab_results"] = res
        st.session_state["lab_summary"] = summary
        st.session_state["percentile_df"] = pct
        prog.progress(100, text="Done")
        status.update(label="Walk-forward lab complete", state="complete", expanded=False)

    if st.session_state["lab_summary"] is not None:
        summary = st.session_state["lab_summary"]
        res = st.session_state["lab_results"]
        pct = st.session_state["percentile_df"]
        st.dataframe(summary, use_container_width=True, hide_index=True)
        if not res.empty:
            front_cols = [
                "StreamKey", "Date", "seed", "Result", "TrueMember",
                "PredictedMember", "Top2_pred", "Top1_Correct", "Needed_Top2", "Miss",
                "Margin", "DecisionMode", "MarginSwap",
                "score_0025", "score_0225", "score_0255", "Fired_Rules"
            ]
            res_view = res[[c for c in front_cols if c in res.columns] + [c for c in res.columns if c not in front_cols]]
        else:
            res_view = res
        st.dataframe(res_view, use_container_width=True, hide_index=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button("Download Lab Summary CSV", summary.to_csv(index=False).encode("utf-8"), "lab_summary__core025_northern_lights_v36.csv", "text/csv", use_container_width=True, on_click="ignore")
        with c2:
            st.download_button("Download Lab Per Event CSV", res.to_csv(index=False).encode("utf-8"), "lab_per_event__core025_northern_lights_v36.csv", "text/csv", use_container_width=True, on_click="ignore")
        with c3:
            st.download_button("Download Percentile CSV", pct.to_csv(index=False).encode("utf-8"), "percentile_list__core025_northern_lights_v36.csv", "text/csv", use_container_width=True, on_click="ignore")
        with c4:
            st.download_button("Download Stream Reduction CSV", stream_diag.to_csv(index=False).encode("utf-8"), "stream_reduction__core025_northern_lights_v36.csv", "text/csv", use_container_width=True, on_click="ignore")

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

### v36 decision-layer correction
- Keeps v34 scoring, promoted-library feature matching, full-truth mining, and stream reduction.
- Removes destructive third-choice GATED_TOP3 hijack.
- Top1 is the highest score and Top2 is the second-highest score.
- Optional cautious Top1/Top2 margin swap is OFF by default for clean testing.

### What this build adds from the 75% v22/v31 path
- Promoted separator library scoring
- FULL TRUTH intelligence layer
- On-run truth-mined separators
- `winner_rate * trait_weight` scoring
- Strong `GATED_TOP3` logic
- 025-only member prediction: `0025`, `0225`, `0255`

### Required files
For daily prediction:
1. Full history
2. Promoted separator library
3. FULL TRUTH file recommended for intelligence layer
4. Last 24h optional

For lab:
Same files, then run Walk-forward Lab.
""")
