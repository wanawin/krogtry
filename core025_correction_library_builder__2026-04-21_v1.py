# BUILD: core025_correction_library_builder__2026-04-21_v1

import pandas as pd
from itertools import combinations
from collections import Counter

# ================================
# LOAD FULL TRUTH FILE
# ================================

def load_truth(file_path):
    df = pd.read_csv(file_path)

    required = ["WinningMember", "Top1_actual", "Top2_actual"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df

# ================================
# FEATURE EXTRACTION
# ================================

def extract_basic_traits(row):
    traits = []

    # Example traits (extendable)
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
# BUILD CORRECTION EVENTS
# ================================

def build_correction_events(df):
    rows = []

    for _, r in df.iterrows():

        # misranking condition
        top1 = r["Top1_actual"]
        top2 = r["Top2_actual"]
        winner = r["WinningMember"]

        if pd.isna(top1) or pd.isna(top2) or pd.isna(winner):
            continue

        traits = extract_basic_traits(r)

        # Only care about misranking
        if top1 != winner and top2 == winner:
            rows.append({
                "traits": traits,
                "winner": winner
            })

    return rows

# ================================
# STACK TRAITS
# ================================

def build_trait_stacks(events, min_support=5):

    counter = {}

    for e in events:
        traits = e["traits"]
        winner = e["winner"]

        # single + combos
        for k in range(1, min(4, len(traits)+1)):
            for combo in combinations(traits, k):
                key = tuple(sorted(combo))

                if key not in counter:
                    counter[key] = {
                        "count": 0,
                        "winners": Counter()
                    }

                counter[key]["count"] += 1
                counter[key]["winners"][winner] += 1

    # build final rules
    rules = []

    for k, v in counter.items():

        support = v["count"]
        if support < min_support:
            continue

        winner_member, win_count = v["winners"].most_common(1)[0]
        win_rate = win_count / support

        # only strong correction signals
        if win_rate < 0.55:
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
# SAVE LIBRARY
# ================================

def save_library(df, name="core025_correction_library.csv"):
    df.to_csv(name, index=False)
    print(f"Saved: {name}")
    return name

# ================================
# MAIN
# ================================

def build_correction_library(file_path):
    df = load_truth(file_path)
    events = build_correction_events(df)
    rules = build_trait_stacks(events)

    if len(rules) == 0:
        print("WARNING: No correction rules found.")
    else:
        print(f"Generated {len(rules)} correction rules")

    return save_library(rules)


# ================================
# RUN
# ================================

if __name__ == "__main__":
    build_correction_library("prepared_full_truth_with_stream_stats_v6.csv")
