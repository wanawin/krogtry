
# BUILD: core025_northern_lights__v49_6_DEBUG

import streamlit as st
import pandas as pd
import time

st.title("Core025 Northern Lights — Straight Backtest DEBUG v49.6")

if "hist" not in st.session_state:
    st.session_state["hist"] = None

uploaded = st.file_uploader("Upload FULL HISTORY")

if uploaded:
    df = pd.read_csv(uploaded, sep=None, engine="python")
    st.session_state["hist"] = df
    st.success(f"Loaded {len(df)} rows")

hist = st.session_state.get("hist")

if hist is None:
    st.warning("Upload FULL HISTORY to begin.")
    st.stop()

st.info(f"Rows: {len(hist)}")

max_events = st.slider("Max Events", 5, 100, 25)

if st.button("Run DEBUG Backtest"):
    progress = st.progress(0)
    log = []

    for i in range(max_events):
        try:
            time.sleep(0.05)
            log.append({"event": i, "status": "ok"})
        except Exception as e:
            log.append({"event": i, "status": "fail", "error": str(e)})
            st.error(f"Failed at event {i}: {e}")
            break

        progress.progress((i+1)/max_events)

    df_log = pd.DataFrame(log)
    st.dataframe(df_log)

    csv = df_log.to_csv(index=False).encode()
    st.download_button("Download Debug Log", csv, "debug_log.csv")
