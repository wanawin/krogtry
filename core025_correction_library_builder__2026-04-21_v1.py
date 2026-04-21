# BUILD: core025_combined_library_builder__2026-04-21_v8_FULL_APP

import streamlit as st
import pandas as pd
from itertools import combinations
from collections import Counter

st.set_page_config(layout="wide")

st.title("Core025 Combined Library Builder (v8)")

st.markdown("""
Upload:
1. FULL TRUTH file (required)
2. Existing separator library (optional)

This will generate a COMBINED library usable in your main app.
""")

# ================================
# FILE INPUTS
# ================================

truth_file = st.file_uploader("Upload FULL TRUTH file (.csv or .txt)", type=["csv","txt"])
separator_file = st.file_uploader("Upload existing separator library (optional)", type=["csv"])

min_support = st.slider("Min support", 3, 20, 5)
min_winrate = st.slider("Min win rate", 0.5, 0.9, 0.55)

# ================================
# TRAIT EXTRACTION
# ================================

def extract_traits(row):
    traits = []

    if "seed_sum" in row:
        traits.append(f"seed_sum_{int(row['seed_sum'])}")

    if "parity" in row:
        traits.append(f"parity_{row['parity']}")

    if "has_pair" in row:
        traits.append(f"has_pair_{int(row['has_pair'])}")

    if "high_count" in row:
        traits.append(f"high_count_{int(row['high_count'])}")

    return traits

# ================================
# BUILD CORRECTION LIBRARY
# ================================

def build_correction(df):

    events = []

    for _, r in df.iterrows():

        if pd.isna(r.get("Top1_actual")) or pd.isna(r.get("Top2_actual")) or pd.isna(r.get("WinningMember")):
            continue

        if r["Top1_actual"] != r["WinningMember"] and r["Top2_actual"] == r["WinningMember"]:

            traits = extract_traits(r)

            events.append({
                "traits": traits,
                "winner": r["WinningMember"]
            })

    counter = {}

    for e in events:
        traits = e["traits"]
        winner = e["winner"]

        for k in range(1, min(4, len(traits)+1)):
            for combo in combinations(traits, k):

                key = tuple(sorted(combo))

                if key not in counter:
                    counter[key] = {"count": 0, "winners": Counter()}

                counter[key]["count"] += 1
                counter[key]["winners"][winner] += 1

    rules = []

    for k, v in counter.items():

        support = v["count"]
        if support < min_support:
            continue

        winner_member, win_count = v["winners"].most_common(1)[0]
        win_rate = win_count / support

        if win_rate < min_winrate:
            continue

        rules.append({
            "pair": "CORRECTION",
            "trait_stack": "|".join(k),
            "winner_member": winner_member,
            "winner_rate": round(win_rate, 4),
            "support": support,
            "pair_gap": 0.0,
            "stack_size": len(k)
        })

    return pd.DataFrame(rules)

# ================================
# RUN
# ================================

if st.button("BUILD COMBINED LIBRARY"):

    if truth_file is None:
        st.error("Upload FULL TRUTH file first")
        st.stop()

    try:
        df_truth = pd.read_csv(truth_file)
    except:
        df_truth = pd.read_csv(truth_file, sep="\t")

    st.success(f"Loaded truth file: {len(df_truth)} rows")

    # Build correction rules
    correction_df = build_correction(df_truth)

    st.write(f"Correction rules generated: {len(correction_df)}")

    # Load separator if present
    if separator_file is not None:
        sep_df = pd.read_csv(separator_file)
        st.write(f"Separator rules loaded: {len(sep_df)}")

        combined = pd.concat([sep_df, correction_df], ignore_index=True)
    else:
        combined = correction_df

    st.write(f"Final combined rules: {len(combined)}")

    st.dataframe(combined.head(20))

    # ================================
    # DOWNLOADS
    # ================================

    st.download_button(
        "Download COMBINED library CSV",
        combined.to_csv(index=False),
        file_name="core025_combined_library.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download COMBINED library TXT",
        combined.to_csv(index=False),
        file_name="core025_combined_library.txt",
        mime="text/plain"
    )
