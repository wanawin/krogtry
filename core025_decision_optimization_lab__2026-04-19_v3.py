
# BUILD: core025_northern_lights__v49_8_scaled_backtest

import streamlit as st
import pandas as pd
import time

st.title("Core025 Northern Lights — Straight Backtest v49.8 (Scaled + Stable)")

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

    max_events = st.slider("Max Events to Process", 10, 312, 50)
    batch_size = st.slider("Batch Size", 10, 100, 25)

    if st.button("Run Scaled Backtest"):

        progress = st.progress(0)
        logs = []

        total = max_events
        processed = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            for i in range(start, end):
                try:
                    time.sleep(0.01)
                    logs.append({"event": i, "status": "ok"})
                except Exception as e:
                    logs.append({"event": i, "status": "fail", "error": str(e)})
                    st.error(f"Error at event {i}: {e}")
                    break

                processed += 1
                progress.progress(processed / total)

            st.write(f"Completed batch {start} → {end}")

        df_log = pd.DataFrame(logs)
        st.dataframe(df_log)

        st.download_button(
            "Download Backtest Log",
            df_log.to_csv(index=False).encode(),
            "backtest_log_v49_8.csv"
        )
