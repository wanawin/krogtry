
# BUILD: core025_northern_lights__v49_7_parsing_fixed

import streamlit as st
import pandas as pd
import time

st.title("Core025 Northern Lights — DEBUG v49.7 (Parsing Fixed)")

def load_history(uploaded):
    try:
        return pd.read_csv(uploaded)
    except:
        try:
            return pd.read_csv(uploaded, sep="\t")
        except:
            try:
                return pd.read_fwf(uploaded)
            except:
                return None

uploaded = st.file_uploader("Upload FULL HISTORY (.txt/.csv)")

if uploaded:
    hist = load_history(uploaded)

    if hist is None or hist.empty:
        st.error("Failed to parse file.")
        st.stop()

    st.success(f"Loaded {len(hist)} rows")

    st.write("Columns detected:", list(hist.columns))

    max_events = st.slider("Max Events", 5, 100, 25)

    if st.button("Run Debug Loop"):
        progress = st.progress(0)
        logs = []

        for i in range(max_events):
            try:
                time.sleep(0.02)
                logs.append({"event": i, "status": "ok"})
            except Exception as e:
                logs.append({"event": i, "status": "fail", "error": str(e)})
                st.error(f"Error at event {i}: {e}")
                break

            progress.progress((i+1)/max_events)

        df_log = pd.DataFrame(logs)
        st.dataframe(df_log)

        st.download_button("Download Debug Log",
                           df_log.to_csv(index=False).encode(),
                           "debug_log_v49_7.csv")
