# BUILD: core025_all3_member_straight_rank_streamlit__2026-05-10_v3
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from core025_all3_engine import (
    BUILD_MARKER,
    load_table,
    normalize_history,
    normalize_lab_events,
    generate_all3,
)

APP_BUILD = "BUILD: core025_all3_member_straight_rank_streamlit__2026-05-10_v3"

st.set_page_config(page_title="Core025 All-3 Straight Generator", layout="wide")

st.title("Core025 All-3-Member Straight Rank Generator")
st.caption(APP_BUILD)
st.info(
    "Upload FULL HISTORY + LAB PER-EVENT or v78 per-event detail, then press the run button. "
    "This is a standalone research generator and does not modify the production app."
)

with st.sidebar:
    st.header("Inputs")
    history_file = st.file_uploader("Upload FULL HISTORY (.txt/.csv/.tsv)", type=["txt", "csv", "tsv"])
    lab_file = st.file_uploader("Upload LAB PER-EVENT or v78 DETAIL (.txt/.csv/.tsv)", type=["txt", "csv", "tsv"])
    depth = st.slider("Max ranked straights per member to export", 5, 24, 12, 1)
    run = st.button("GENERATE ALL-3-MEMBER FILES", type="primary", use_container_width=True)

if not history_file or not lab_file:
    st.warning("Upload both files to continue.")
    st.stop()

def save_upload(uploaded, folder: Path) -> Path:
    path = folder / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return path

if not run:
    st.success("Files are uploaded. Press the red GENERATE button in the sidebar.")
    st.stop()

try:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        hist_path = save_upload(history_file, tmpdir)
        lab_path = save_upload(lab_file, tmpdir)
        outdir = tmpdir / "outputs"

        st.write("Loading files...")
        hist = normalize_history(load_table(hist_path))
        lab = normalize_lab_events(load_table(lab_path), hist)

        c1, c2, c3 = st.columns(3)
        c1.metric("History rows", f"{len(hist):,}")
        c2.metric("Lab events", f"{len(lab):,}")
        c3.metric("Depth/member", depth)

        st.write("Generating all-3-member straight ranks. This can take a few minutes.")
        with st.spinner("Generating all-3-member files..."):
            generate_all3(lab, hist, int(depth), outdir)

        ranked_path = outdir / "all_3_members_ranked_per_event__core025_v1.csv"
        event_path = outdir / "all_3_members_event_summary__core025_v1.csv"
        selector_path = outdir / "selector_baseline_summary__core025_v1.csv"

        ranked_bytes = ranked_path.read_bytes()
        event_bytes = event_path.read_bytes()
        selector_bytes = selector_path.read_bytes()

        selector = pd.read_csv(selector_path)
        st.success("Generation complete.")
        st.subheader("Selector Baseline Summary")
        st.dataframe(selector, use_container_width=True, hide_index=True)

        st.download_button(
            "Download all_3_members_ranked_per_event__core025_v1.csv",
            ranked_bytes,
            file_name="all_3_members_ranked_per_event__core025_v1.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download all_3_members_event_summary__core025_v1.csv",
            event_bytes,
            file_name="all_3_members_event_summary__core025_v1.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download selector_baseline_summary__core025_v1.csv",
            selector_bytes,
            file_name="selector_baseline_summary__core025_v1.csv",
            mime="text/csv",
            use_container_width=True,
        )

except Exception as e:
    st.error("Generator failed.")
    st.exception(e)
