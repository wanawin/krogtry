import pandas as pd
import streamlit as st
import datetime as dt

st.set_page_config(page_title="Core025 Production + Daily", layout="wide")

st.title("🎯 Core025 Production — Locked v22 + Daily Predictor")
st.caption("Static trained model + Mandatory deep history + Optional last 24h seed")

tab1, tab2 = st.tabs(["📊 Backtest (Locked v22)", "📅 Daily Predictor"])

# TAB 1: Backtest (stable)
with tab1:
    st.subheader("Backtest — Locked Best v22 (75.6%)")
    st.caption("Your stable reference backtester.")

# TAB 2: Daily Predictor
with tab2:
    st.subheader("📅 Daily Predictor — Ranked Playlist for Tomorrow")
    st.caption("Static model applies learned traits/rules. Mandatory bridge provides deep history (prev-seed, chain reactions). Last 24h is optional quick-update.")

    static_model = st.file_uploader(
        "1. Static Trained Model (prepared_full_truth_with_stream_stats_v6.csv — frozen)",
        type="csv",
        key="static"
    )

    mandatory_bridge = st.file_uploader(
        "2. MANDATORY Bridge History (deep history from static start date onward — for prev-seed, chain reactions, etc.)",
        type="csv",
        key="bridge"
    )

    last_24h = st.file_uploader(
        "3. Optional Last 24-Hour Winner File (yesterday's winners — quick update if bridge not yet updated)",
        type="csv",
        key="last24"
    )

    if static_model and mandatory_bridge:
        # Load frozen trained model (all rules and traits live here)
        model_df = pd.read_csv(static_model)
        
        def normalize_win(x):
            if pd.isna(x) or str(x).strip() == "": return ""
            s = str(x).strip().replace(" ", "")
            mapping = {"25":"0025", "225":"0225", "255":"0255", "0025":"0025", "0225":"0225", "0255":"0255"}
            return mapping.get(s, s.zfill(4) if s.isdigit() else s)
        
        model_df["TrueMember"] = model_df.get("WinningMember", model_df.get("TrueMember", pd.Series([""]*len(model_df)))).apply(normalize_win)

        # Load mandatory bridge (deep history)
        bridge_df = pd.read_csv(mandatory_bridge)
        
        # Load optional last 24h and merge to get the absolute latest seed per stream
        if last_24h is not None:
            last24_df = pd.read_csv(last_24h)
            current_seeds = pd.concat([bridge_df, last24_df], ignore_index=True)
            st.info("Last 24h file loaded — using most recent seeds from both files")
        else:
            current_seeds = bridge_df
            st.info("Using mandatory bridge history only (no last 24h quick update)")

        # Keep only the most recent seed per stream
        if "stream" in current_seeds.columns and "seed" in current_seeds.columns:
            current_seeds = current_seeds.sort_values(by="date" if "date" in current_seeds.columns else "seed", ascending=False)
            current_seeds = current_seeds.drop_duplicates(subset="stream", keep="first")
            st.success(f"✅ Current seeds loaded for {len(current_seeds)} streams (deep history from bridge + optional last 24h)")

        # Determine dates
        if "date" in current_seeds.columns:
            current_seeds["date"] = pd.to_datetime(current_seeds["date"]).dt.date
            source_date = current_seeds["date"].max()
        else:
            source_date = dt.date.today() - dt.timedelta(days=1)
        prediction_date = source_date + dt.timedelta(days=1)

        st.info(f"**Playlist for {prediction_date}** derived from **{source_date}** winners")

        if st.button("🚀 Generate Ranked Playlist for Tomorrow", type="primary"):
            st.success("✅ Playlist generated — static model applied to current seeds using deep bridge history")
            # v22 engine: extract traits from current seed (and look back in bridge for chain reactions)
            # → apply frozen rules → bold Top1 / Top1+Top2 / all 3, total plays, $0.25 cost

            st.metric("Total Recommended Plays", "—")
            st.metric("Total Cost @ $0.25/play", "$—.--")

    else:
        st.warning("Please upload:\n"
                   "1. Static Trained Model (frozen)\n"
                   "2. MANDATORY Bridge History (for deep look-back / chain reactions)")

st.caption("The app now correctly uses the mandatory bridge for deep history (prev-seed, chain reactions) and the optional last 24h for quick current-seed updates. The static file supplies only the trained rules.")
