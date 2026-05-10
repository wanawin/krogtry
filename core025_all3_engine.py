#!/usr/bin/env python3
# BUILD: core025_all3_member_straight_rank_generator__2026-05-10_v1
"""
Standalone Core025 all-3-member straight rank generator.

Purpose
-------
Generate ranked straight permutations for ALL THREE Core025 members
(0025, 0225, 0255) for every event in a lab_per_event file.

This is intentionally OUTSIDE the production Streamlit app so mining/testing
can happen without risking the working box engine.

Inputs
------
--history : full lottery history file used by the production app (.csv/.txt/.tsv)
--lab     : lab_per_event file from the stable box/walk-forward app (.csv/.txt/.tsv)
--outdir  : output folder
--depth   : max ranked straights per member to export, default 12

Outputs
-------
1. all_3_members_ranked_per_event__core025_v1.csv
2. all_3_members_event_summary__core025_v1.csv
3. selector_baseline_summary__core025_v1.csv

Scoring stack matches the recovered straight engine:
25% member + 25% position + 20% ordered_pair + 15% seed_transition + 10% stream_order + 5% recent_order
"""

from __future__ import annotations

import argparse
import re
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

BUILD_MARKER = "BUILD: core025_all3_member_straight_rank_generator__2026-05-10_v1"
MEMBERS = ["0025", "0225", "0255"]


def load_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(path, dtype=str, low_memory=False)
    if name.endswith((".txt", ".tsv")):
        try:
            df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
            if df.shape[1] >= 4:
                return df
        except Exception:
            pass
        return pd.read_csv(path, sep="\t", header=None, dtype=str, low_memory=False)
    raise ValueError(f"Unsupported file type: {path}")


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
    return "".join(digits[:4]) if len(digits) >= 4 else ""


def box_key(s: str) -> str:
    s = extract_pick4_digits(s) or str(s)
    return "".join(sorted(re.sub(r"\D", "", s).zfill(4)[-4:]))


def normalize_member(x) -> str:
    s = re.sub(r"\D", "", str(x or "").strip())
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


def digits4(seed: str) -> List[int]:
    s = extract_pick4_digits(seed)
    if len(s) != 4:
        s = re.sub(r"\D", "", str(seed)).zfill(4)[-4:]
    return [int(ch) for ch in s] if len(s) == 4 else [0, 0, 0, 0]


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
    out["seed"] = out[seed_col].astype(str).apply(lambda x: extract_pick4_digits(x) or re.sub(r"\D", "", str(x)).zfill(4)[-4:])
    out["seed_sum"] = sums
    out["seed_spread"] = spreads
    out["seed_even_cnt"] = [sum(1 for x in d if x % 2 == 0) for d in ds]
    out["seed_high_cnt"] = [sum(1 for x in d if x >= 5) for d in ds]
    out["seed_low_cnt"] = [sum(1 for x in d if x <= 4) for d in ds]
    out["parity_pattern"] = ["".join("E" if x % 2 == 0 else "O" for x in d) for d in ds]
    out["highlow_pattern"] = ["".join("H" if x >= 5 else "L" for x in d) for d in ds]
    out["structure"] = [structure_4(d) for d in ds]
    out["unique"] = [len(set(d)) for d in ds]
    out["max_rep"] = [max(pd.Series(d).value_counts().tolist()) for d in ds]
    out["consec_links"] = [consec_links_count(d) for d in ds]
    out["pair"] = [1 if max(pd.Series(d).value_counts().tolist()) >= 2 else 0 for d in ds]
    out["sum_bucket"] = [sum_bucket(int(v)) for v in sums]
    out["spread_bucket"] = [spread_bucket(int(v)) for v in spreads]
    for digit in range(10):
        out[f"cnt{digit}"] = [d.count(digit) for d in ds]
        out[f"has{digit}"] = [1 if digit in d else 0 for d in ds]
    return out


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
            raise ValueError("History file needs Results/Result column.")
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
    out["StreamKey"] = out.apply(lambda r: canonical_stream(r["State"], r["Game"]), axis=1)
    out["Core025Member"] = out["Result"].apply(result_to_core025_member)
    return out.sort_values(["StreamKey", "Date"]).reset_index(drop=True)


def normalize_lab_events(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in ["StreamKey", "Date", "seed", "Result"]:
        if c not in df.columns:
            raise ValueError(f"Lab per-event file missing required column: {c}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["seed"] = df["seed"].astype(str).apply(lambda x: extract_pick4_digits(x) or re.sub(r"\D", "", str(x)).zfill(4)[-4:])
    df["Result"] = df["Result"].astype(str).apply(extract_pick4_digits)
    if "TrueMember" not in df.columns:
        df["TrueMember"] = df["Result"].apply(result_to_core025_member)
    else:
        df["TrueMember"] = df["TrueMember"].apply(normalize_member)
    for c in ["PredictedMember", "Top2_pred", "ThirdMember"]:
        if c in df.columns:
            df[c] = df[c].apply(normalize_member)
        else:
            df[c] = ""
    return enrich_seed_features(df, "seed").reset_index(drop=True)


def duplicate_position_pattern(perm: str) -> str:
    counts = pd.Series(list(str(perm))).value_counts().to_dict()
    dup_digits = [d for d, c in counts.items() if c > 1]
    if not dup_digits:
        return "none"
    d = sorted(dup_digits)[0]
    return "-".join(str(i + 1) for i, ch in enumerate(str(perm)) if ch == d)


def ordered_pairs_4(perm: str) -> List[str]:
    s4 = str(perm).zfill(4)[-4:]
    return [s4[:2], s4[1:3], s4[2:4]]


def unique_straight_permutations(member: str) -> List[str]:
    member = normalize_member(member)
    return sorted({"".join(p) for p in permutations(member, 4)}) if member in MEMBERS else []


def build_straight_training_events(hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stream, g in hist.groupby("StreamKey"):
        g = g.sort_values("Date").reset_index(drop=True)
        for i in range(1, len(g)):
            member = result_to_core025_member(g.loc[i, "Result"])
            if member in MEMBERS:
                rows.append({
                    "StreamKey": stream,
                    "Date": g.loc[i, "Date"],
                    "seed": g.loc[i - 1, "Result"],
                    "TrueMember": member,
                    "StraightResult": extract_pick4_digits(g.loc[i, "Result"]),
                    "Result": extract_pick4_digits(g.loc[i, "Result"]),
                })
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events = enrich_seed_features(events, "seed")
    events["DupPattern"] = events["StraightResult"].apply(duplicate_position_pattern)
    events["Pair12"] = events["StraightResult"].astype(str).str[:2]
    events["Pair23"] = events["StraightResult"].astype(str).str[1:3]
    events["Pair34"] = events["StraightResult"].astype(str).str[2:4]
    for pos in range(4):
        events[f"pos{pos+1}"] = events["StraightResult"].astype(str).str[pos]
    return events


def _rate_with_support(df: pd.DataFrame, mask, min_support: int = 1) -> Tuple[float, int]:
    support = int(len(df))
    if support < min_support or support <= 0:
        return 0.0, support
    return float(pd.Series(mask).sum()) / support, support


def straight_member_score(row: pd.Series, member: str) -> float:
    member = normalize_member(member)
    scores = {}
    for m in MEMBERS:
        col = f"score_{m}"
        if col in row.index:
            scores[m] = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").fillna(0).iloc[0]
    mx = max(scores.values()) if scores else 0.0
    if mx > 0 and member in scores:
        return max(0.0, min(1.0, scores[member] / mx))
    if member == normalize_member(row.get("PredictedMember", "")):
        return 1.0
    if member == normalize_member(row.get("Top2_pred", "")):
        return 0.65
    if member == normalize_member(row.get("ThirdMember", "")):
        return 0.35
    return 0.0


def score_perm(row: pd.Series, member: str, perm: str, events: pd.DataFrame) -> Dict:
    member = normalize_member(member)
    perm = str(perm).zfill(4)[-4:]
    stream = str(row.get("StreamKey", ""))
    member_events = events[events["TrueMember"].eq(member)].copy()
    stream_events = member_events[member_events["StreamKey"].eq(stream)].copy()
    trait_events = member_events[(member_events["sum_bucket"].astype(str).eq(str(row.get("sum_bucket", "")))) & (member_events["spread_bucket"].astype(str).eq(str(row.get("spread_bucket", ""))))].copy()
    shape_events = member_events[(member_events["parity_pattern"].astype(str).eq(str(row.get("parity_pattern", "")))) & (member_events["highlow_pattern"].astype(str).eq(str(row.get("highlow_pattern", ""))))].copy()
    exact_seed = member_events[member_events["seed"].astype(str).eq(str(row.get("seed", "")))].copy()

    pos_rates, pos_supports = [], []
    for i, digit in enumerate(perm):
        rg, sg = _rate_with_support(member_events, member_events[f"pos{i+1}"].astype(str).eq(digit), 1)
        rs, ss = _rate_with_support(stream_events, stream_events[f"pos{i+1}"].astype(str).eq(digit), 2)
        rt, st = _rate_with_support(trait_events, trait_events[f"pos{i+1}"].astype(str).eq(digit), 3)
        pos_rates.append((0.55 * rg) + (0.30 * rs) + (0.15 * rt))
        pos_supports.append(sg + ss + st)
    position_score = sum(pos_rates) / 4 if pos_rates else 0.0

    pair_rates, pair_supports = [], []
    for pair, col in zip(ordered_pairs_4(perm), ["Pair12", "Pair23", "Pair34"]):
        rg, sg = _rate_with_support(member_events, member_events[col].astype(str).eq(pair), 1)
        rs, ss = _rate_with_support(stream_events, stream_events[col].astype(str).eq(pair), 2)
        rsh, ssh = _rate_with_support(shape_events, shape_events[col].astype(str).eq(pair), 3)
        pair_rates.append((0.55 * rg) + (0.25 * rs) + (0.20 * rsh))
        pair_supports.append(sg + ss + ssh)
    ordered_pair_score = sum(pair_rates) / 3 if pair_rates else 0.0

    rex, sex = _rate_with_support(exact_seed, exact_seed["StraightResult"].astype(str).eq(perm), 1)
    rtr, strt = _rate_with_support(trait_events, trait_events["StraightResult"].astype(str).eq(perm), 3)
    rsh, ssh = _rate_with_support(shape_events, shape_events["StraightResult"].astype(str).eq(perm), 3)
    seed_transition_score = (0.50 * rex) + (0.30 * rtr) + (0.20 * rsh)

    first_rate, first_support = _rate_with_support(stream_events, stream_events["pos1"].astype(str).eq(perm[0]), 2)
    last_rate, last_support = _rate_with_support(stream_events, stream_events["pos4"].astype(str).eq(perm[3]), 2)
    stream_order_score = (first_rate + last_rate) / 2 if (first_support + last_support) > 0 else 0.0

    recent = stream_events.sort_values("Date").tail(20)
    rr, rsup = _rate_with_support(recent, recent["StraightResult"].astype(str).eq(perm), 1)
    rf, rfs = _rate_with_support(recent, recent["pos1"].astype(str).eq(perm[0]), 2)
    rb, rbs = _rate_with_support(recent, recent["pos4"].astype(str).eq(perm[3]), 2)
    recent_order_score = (0.50 * rr) + (0.25 * rf) + (0.25 * rb)

    dup = duplicate_position_pattern(perm)
    rdg, sdg = _rate_with_support(member_events, member_events["DupPattern"].astype(str).eq(dup), 1)
    rds, sds = _rate_with_support(stream_events, stream_events["DupPattern"].astype(str).eq(dup), 2)
    repeat_placement_score = (0.70 * rdg) + (0.30 * rds)

    member_conf = straight_member_score(row, member)
    total = (member_conf * 0.25) + (position_score * 0.25) + (ordered_pair_score * 0.20) + (seed_transition_score * 0.15) + (stream_order_score * 0.10) + (recent_order_score * 0.05)
    evidence = int(sum(pos_supports) + sum(pair_supports) + sex + strt + ssh + first_support + last_support + rsup + rfs + rbs + sdg + sds)

    return {
        "StraightPermutation": perm,
        "MemberConfidenceScore": round(member_conf * 100, 2),
        "PositionScore": round(position_score * 100, 2),
        "OrderedPairScore": round(ordered_pair_score * 100, 2),
        "SeedTransitionScore": round(seed_transition_score * 100, 2),
        "StreamOrderScore": round(stream_order_score * 100, 2),
        "RecentOrderScore": round(recent_order_score * 100, 2),
        "RepeatPlacementScore": round(repeat_placement_score * 100, 2),
        "StraightConfidenceScore": round(total * 100, 2),
        "EvidenceSupport": evidence,
        "DupPattern": dup,
        "StraightScoreFormula": "25% member + 25% position + 20% ordered_pair + 15% seed_transition + 10% stream_order + 5% recent_order",
    }


def generate_all3(lab: pd.DataFrame, hist: pd.DataFrame, depth: int, outdir: Path) -> None:
    events = build_straight_training_events(hist)
    if events.empty:
        raise ValueError("No Core025 straight training events found in history.")

    ranked_parts = []
    event_rows = []
    n = len(lab)
    for idx, row in lab.iterrows():
        if idx % 25 == 0:
            print(f"Scoring event {idx + 1}/{n}...")
        true_straight = extract_pick4_digits(row.get("Result", ""))
        true_member = normalize_member(row.get("TrueMember", result_to_core025_member(true_straight)))
        event_ranked = []
        for member in MEMBERS:
            rows = []
            for perm in unique_straight_permutations(member):
                rows.append({
                    "EventIndex": idx,
                    "Date": row.get("Date", ""),
                    "StreamKey": row.get("StreamKey", ""),
                    "SingleRow": row.get("SingleRow", ""),
                    "StreamRank": row.get("StreamRank", ""),
                    "RowPercentile": row.get("RowPercentile", ""),
                    "StreamTier": row.get("StreamTier", ""),
                    "seed": row.get("seed", ""),
                    "Result": true_straight,
                    "TrueMember": true_member,
                    "PredictedMember": normalize_member(row.get("PredictedMember", "")),
                    "Top2_pred": normalize_member(row.get("Top2_pred", "")),
                    "ThirdMember": normalize_member(row.get("ThirdMember", "")),
                    "PlayType": row.get("PlayType", ""),
                    "Top2Decision": row.get("Top2Decision", ""),
                    "Top2RiskScore": row.get("Top2RiskScore", ""),
                    "Top2ToTop1Ratio": row.get("Top2ToTop1Ratio", ""),
                    "Margin": row.get("Margin", ""),
                    "RowVolatilityRate": row.get("RowVolatilityRate", ""),
                    "RowTop1Rate": row.get("RowTop1Rate", ""),
                    "RowTop2Rate": row.get("RowTop2Rate", ""),
                    "RowTop3Rate": row.get("RowTop3Rate", ""),
                    "RowPlayType": row.get("RowPlayType", row.get("PlayType", "")),
                    "RowPlayTypeReason": row.get("RowPlayTypeReason", ""),
                    "Top3Rescue": row.get("Top3Rescue", ""),
                    "Top3RescueReasons": row.get("Top3RescueReasons", ""),
                    "RankedMember": member,
                    **score_perm(row, member, perm, events),
                })
            mdf = pd.DataFrame(rows).sort_values(["StraightConfidenceScore", "EvidenceSupport"], ascending=[False, False]).reset_index(drop=True)
            mdf["MemberStraightRank"] = mdf.index + 1
            mdf["MemberAnyRankHit"] = mdf["StraightPermutation"].astype(str).str.zfill(4).eq(true_straight).astype(int)
            mdf["MemberTop1Hit"] = ((mdf["MemberAnyRankHit"] == 1) & (mdf["MemberStraightRank"] <= 1)).astype(int)
            mdf["MemberTop3Hit"] = ((mdf["MemberAnyRankHit"] == 1) & (mdf["MemberStraightRank"] <= 3)).astype(int)
            mdf["MemberTop5Hit"] = ((mdf["MemberAnyRankHit"] == 1) & (mdf["MemberStraightRank"] <= 5)).astype(int)
            ranked_parts.append(mdf[mdf["MemberStraightRank"] <= depth].copy())
            event_ranked.append(mdf)

        ev_all = pd.concat(event_ranked, ignore_index=True)
        summary = {
            "EventIndex": idx,
            "Date": row.get("Date", ""),
            "StreamKey": row.get("StreamKey", ""),
            "SingleRow": row.get("SingleRow", ""),
            "StreamRank": row.get("StreamRank", ""),
            "RowPercentile": row.get("RowPercentile", ""),
            "seed": row.get("seed", ""),
            "Result": true_straight,
            "TrueMember": true_member,
            "PredictedMember": normalize_member(row.get("PredictedMember", "")),
            "Top2_pred": normalize_member(row.get("Top2_pred", "")),
            "ThirdMember": normalize_member(row.get("ThirdMember", "")),
            "PlayType": row.get("PlayType", ""),
            "Top2Decision": row.get("Top2Decision", ""),
            "Top2RiskScore": row.get("Top2RiskScore", ""),
            "RowVolatilityRate": row.get("RowVolatilityRate", ""),
            "RowPlayTypeReason": row.get("RowPlayTypeReason", ""),
        }
        for member in MEMBERS:
            sub = ev_all[ev_all["RankedMember"].eq(member)].copy()
            hit_rows = sub[sub["StraightPermutation"].astype(str).str.zfill(4).eq(true_straight)]
            summary[f"{member}_TopStraight"] = sub.iloc[0]["StraightPermutation"] if not sub.empty else ""
            summary[f"{member}_TopScore"] = sub.iloc[0]["StraightConfidenceScore"] if not sub.empty else ""
            summary[f"{member}_HitRank"] = int(hit_rows["MemberStraightRank"].min()) if not hit_rows.empty else None
            for d in [1, 3, 5]:
                val = summary[f"{member}_HitRank"]
                summary[f"{member}_Top{d}Hit"] = int(pd.notna(val) and int(val) <= d)
        def hit_for(member: str, d: int) -> int:
            member = normalize_member(member)
            val = summary.get(f"{member}_HitRank")
            return int(member in MEMBERS and pd.notna(val) and int(val) <= d)
        for name, member in [("AlwaysTop1", summary["PredictedMember"]), ("AlwaysTop2", summary["Top2_pred"]), ("AlwaysThird", summary["ThirdMember"]), ("TrueMemberOracle", true_member)]:
            summary[f"{name}_Member"] = member
            for d in [1, 3, 5]:
                summary[f"{name}_Top{d}Hit"] = hit_for(member, d)
        event_rows.append(summary)

    ranked = pd.concat(ranked_parts, ignore_index=True)
    summary_df = pd.DataFrame(event_rows)
    selector_rows = []
    total = len(summary_df)
    for selector in ["AlwaysTop1", "AlwaysTop2", "AlwaysThird", "TrueMemberOracle"]:
        selector_rows.append({
            "Selector": selector,
            "FullUniverseTotal": total,
            "Top1Hits": int(summary_df[f"{selector}_Top1Hit"].sum()),
            "Top3Hits": int(summary_df[f"{selector}_Top3Hit"].sum()),
            "Top5Hits": int(summary_df[f"{selector}_Top5Hit"].sum()),
            "Top5Pct": round(int(summary_df[f"{selector}_Top5Hit"].sum()) / total * 100, 2) if total else 0,
        })
    selector_summary = pd.DataFrame(selector_rows)

    outdir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(outdir / "all_3_members_ranked_per_event__core025_v1.csv", index=False)
    summary_df.to_csv(outdir / "all_3_members_event_summary__core025_v1.csv", index=False)
    selector_summary.to_csv(outdir / "selector_baseline_summary__core025_v1.csv", index=False)
    print("\nDone.")
    print(selector_summary.to_string(index=False))
    print(f"\nSaved outputs to: {outdir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Core025 all-3-member straight rank files.")
    p.add_argument("--history", required=True, help="Full history file (.csv/.txt/.tsv)")
    p.add_argument("--lab", required=True, help="Lab per-event file (.csv/.txt/.tsv)")
    p.add_argument("--outdir", default="core025_all3_outputs", help="Output folder")
    p.add_argument("--depth", type=int, default=12, help="Max ranked straights per member to export")
    args = p.parse_args()

    print(BUILD_MARKER)
    hist = normalize_history(load_table(Path(args.history)))
    lab = normalize_lab_events(load_table(Path(args.lab)))
    print(f"History rows: {len(hist):,}")
    print(f"Lab events:   {len(lab):,}")
    generate_all3(lab, hist, int(args.depth), Path(args.outdir))


if __name__ == "__main__":
    main()
