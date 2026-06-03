import pandas as pd
import streamlit as st
import datetime as dt
import numpy as np

st.set_page_config(page_title="Core025 Northern Lights v146", layout="wide")

BUILD_MARKER = "BUILD: core025_northern_lights__2026-06-03_v146_HYBRID_REALISTIC"

st.title("Core025 Northern Lights — 025 Live + Lab")
st.caption(BUILD_MARKER)

def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith(('.txt', '.tsv')):
            return pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load {uploaded_file.name}: {str(e)}")
        return None

def parse_date_column(df):
    if df is None or df.empty:
        return df
    date_col = next((c for c in ['date', 'Date', 'DrawDate', 'draw_date'] if c in df.columns), None)
    if not date_col:
        return df
    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']:
        try:
            df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors='coerce').dt.date
            if df[date_col].notna().any():
                return df
        except:
            pass
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    return df

# ====================== IMPROVED HYBRID v146 ======================
def build_true_hybrid_rank_actual_cost(playlist):
    if playlist is None or playlist.empty:
        return pd.DataFrame()
    
    out = playlist.copy()
    
    def safe_num(col, default=0.0):
        if col in out.columns:
            return pd.to_numeric(out[col], errors='coerce').fillna(default)
        return pd.Series(default, index=out.index)
    
    # Dynamic components with heavy fallbacks
    top1 = safe_num('Top1_score')
    if top1.std() == 0:
        top1 = safe_num('ModelConfidenceScore')
    if top1.std() == 0:
        top1 = safe_num('StreamRank', 1)  # fallback to rank itself
    
    gap = safe_num('gap')
    if gap.std() == 0:
        gap = safe_num('Margin')
    
    ratio = safe_num('ratio')
    if ratio.std() == 0:
        ratio = safe_num('Top2ToTop1Ratio')
    
    top23 = safe_num('Top2_score') + safe_num('Top3_score')
    
    def safe_rank(s, ascending=True):
        if s.nunique() <= 1:
            return pd.Series(range(1, len(s)+1), index=s.index)
        return s.rank(method='first', ascending=ascending)
    
    out['DynamicScore'] = (
        1.15 * safe_rank(top1, ascending=False) +
        1.00 * safe_rank(gap, ascending=False) +
        0.85 * safe_rank(ratio) +
        0.70 * safe_rank(top23)
    )
    out['DynamicRank'] = safe_rank(out['DynamicScore'], ascending=False).astype(int)
    
    # SingleRow fallback chain
    sr = safe_num('SingleRow')
    if sr.std() == 0:
        sr = safe_num('RowPercentile')
    if sr.std() == 0:
        sr = safe_num('StreamRank')
    if sr.std() == 0:
        sr = pd.Series(range(1, len(out)+1), index=out.index)
    
    out['SingleRowHistoricalRank'] = safe_rank(sr, ascending=True).astype(int)
    out['DuePressureRank'] = safe_rank(sr, ascending=False).astype(int)
    
    out['HybridScore'] = 0.50 * out['DynamicRank'] + 0.35 * out['SingleRowHistoricalRank'] + 0.15 * out['DuePressureRank']
    out['HybridFinalRank'] = safe_rank(out['HybridScore'], ascending=True).astype(int)
    
    # Robust Actual Play Count
    play_col = next((c for c in ['ActualBoxPlay', 'ActualBoxPlayDisplay', 'RecommendedPlay', 'PredictedMember', 'Top1'] if c in out.columns), None)
    if play_col:
        out['ActualMembersToPlay'] = out[play_col].fillna('0025').astype(str)
    else:
        out['ActualMembersToPlay'] = '0025'
    
    # Limit to realistic 1-3 plays per stream
    out['ActualPlayCount'] = out['ActualMembersToPlay'].str.count(r'0025|0225|0255').clip(lower=1, upper=3).astype(int)
    out['Action'] = out['ActualPlayCount'].map({1:'TOP1', 2:'TOP2', 3:'TOP3'})
    out['RowCostDisplay'] = (out['ActualPlayCount'] * 0.25).map(lambda x: f"${x:.2f}")
    
    # Final sort + running totals
    out = out.sort_values('HybridFinalRank').reset_index(drop=True)
    out['PlayOrder'] = range(1, len(out)+1)
    out['RunningPlayTotal'] = out['ActualPlayCount'].cumsum()
    out['RunningCostDisplay'] = (out['RunningPlayTotal'] * 0.25).map(lambda x: f"${x:.2f}")
    
    # Column order
    front = ['PlayOrder', 'HybridFinalRank', 'HybridScore', 'StreamKey', 'State', 'Game',
             'Action', 'ActualMembersToPlay', 'ActualPlayCount', 'RowCostDisplay',
             'RunningPlayTotal', 'RunningCostDisplay', 'OldStreamRank', 'DynamicRank',
             'SingleRowHistoricalRank', 'DuePressureRank']
    front = [c for c in front if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]

# ====================== UI ======================
tab1, tab2 = st.tabs(["📊 Backtest", "📅 Daily Predictor"])

with tab2:
    st.subheader("📅 Daily Predictor — Ranked Playlist for Tomorrow")
    static_model = st.file_uploader("Static Trained Model", type=["csv","txt","tsv"], key="daily_static")
    separator_library = st.file_uploader("Separator Traits Library", type=["csv","txt","tsv"], key="daily_separators")
    mandatory_bridge = st.file_uploader("MANDATORY Bridge History (deep history)", type=["csv","txt","tsv"], key="daily_bridge")
    last_24h = st.file_uploader("Optional Last 24-Hour Winner File", type=["csv","txt","tsv"], key="daily_24h")

    if static_model and separator_library and mandatory_bridge:
        model_df = load_file(static_model)
        bridge_df = load_file(mandatory_bridge)
        
        if last_24h is not None:
            last24_df = load_file(last_24h)
            current_seeds = pd.concat([bridge_df, last24_df], ignore_index=True)
        else:
            current_seeds = bridge_df

        current_seeds = parse_date_column(current_seeds)
        
        source_date = current_seeds['date'].max() if current_seeds is not None and 'date' in current_seeds.columns and current_seeds['date'].notna().any() else dt.date.today() - dt.timedelta(days=1)
        prediction_date = source_date + dt.timedelta(days=1)

        st.info(f"**Most recent history date:** {source_date} → **Predicting playlist for:** {prediction_date}")

        if st.button("🚀 Generate Ranked Playlist for Tomorrow", type="primary"):
            hybrid_table = build_true_hybrid_rank_actual_cost(current_seeds)
            
            st.subheader("FINAL PLAY THIS TODAY — TRUE HYBRID RANKED PLAYLIST")
            st.dataframe(hybrid_table, use_container_width=True, hide_index=True)
            
            total_plays = int(hybrid_table['ActualPlayCount'].sum())
            total_cost = total_plays * 0.25
            st.metric("Total Plays", total_plays)
            st.metric("Total Cost @ $0.25/play", f"${total_cost:.2f}")

            csv_data = hybrid_table.to_csv(index=False).encode('utf-8')
            txt_data = hybrid_table.to_string(index=False)
            
            st.download_button("📥 Download Playlist as CSV", csv_data, f"playlist_{prediction_date}.csv", "text/csv")
            st.download_button("📥 Download Playlist as TXT", txt_data, f"playlist_{prediction_date}.txt", "text/plain")

    else:
        st.warning("Upload Static Model + Separator Library + Mandatory Bridge History")

st.caption("v146: Hybrid ranking with strong fallbacks + realistic play count (max 3 per stream).")
