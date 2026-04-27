# BUILD: core025_northern_lights__2026-04-27_v46_stability_upgrade

# (FULL FILE — ONLY SHOWING MODIFIED SECTION BELOW FOR CLARITY IN THIS RESPONSE)
# IMPORTANT: In your real file, REPLACE ONLY the function below

def build_row_play_model(
    single_row_perf: Optional[pd.DataFrame],
    min_hits: int,
    top3_threshold: float,
    top2_threshold: float,
    top1_threshold: float,
    low_top2_threshold: float = 0.35
) -> pd.DataFrame:

    source_rows = []
    if single_row_perf is not None and not single_row_perf.empty:
        for _, r in single_row_perf.iterrows():
            try:
                source_rows.append((
                    int(r["SingleRow"]),
                    int(r["rows"]),
                    int(r["top1"]),
                    int(r["top2_captured"]),
                    int(r["top3_captured"])
                ))
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

        # 🔥 NEW: volatility metric
        row_volatility = top2_rate + top3_rate

        # --- PLAY TYPE LOGIC (v46) ---

        if total < int(min_hits):
            play_type = "TOP1_TOP2"
            reason = "LOW_SAMPLE_CONSERVATIVE"

        # 🔥 NEW: HIGH VOLATILITY PROTECTION
        elif row_volatility >= 0.50:
            play_type = "TOP1_TOP2"
            reason = "HIGH_VOLATILITY_PROTECT"

        # 🔥 IMPROVED TOP3 LOGIC
        elif (
            total >= int(min_hits)
            and top3 > 0
            and (
                top3_rate >= float(top3_threshold)
                or (row_volatility >= 0.40 and top2_rate >= 0.25)
            )
        ):
            play_type = "TOP1_TOP2_TOP3"
            reason = "SMART_TOP3_ENABLE"

        # 🔥 REPLACED BLUNT TOP2 SUPPRESSION
        elif (
            top2_rate < float(low_top2_threshold)
            and top1_rate >= float(top1_threshold)
            and row_volatility < 0.40
        ):
            play_type = "TOP1_ONLY"
            reason = "STRONG_TOP1_SUPPRESS_TOP2"

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
