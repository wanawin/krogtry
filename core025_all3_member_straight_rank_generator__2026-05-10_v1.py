# BUILD: core025_all3_member_straight_rank_streamlit_SINGLEFILE__2026-05-11_v6_SMART_TXT_DELIMITER
from __future__ import annotations
import io, re
from itertools import permutations
import pandas as pd
import streamlit as st

BUILD_MARKER='BUILD: core025_all3_member_straight_rank_streamlit_SINGLEFILE__2026-05-11_v6_SMART_TXT_DELIMITER'
MEMBERS=['0025','0225','0255']

def extract_pick4_digits(x):
    if x is None: return ''
    s=str(x)
    m=re.search(r'(?<!\d)(\d{4})(?!\d)',s)
    if m: return m.group(1)
    d=re.findall(r'\d',s)
    return ''.join(d[:4]) if len(d)>=4 else ''

def box_key(x):
    s=extract_pick4_digits(x)
    return ''.join(sorted(s)) if len(s)==4 else ''

def normalize_member(x):
    s=re.sub(r'\D','',str(x or ''))
    if s in {'25','025','0025'}: return '0025'
    if s in {'225','0225'}: return '0225'
    if s in {'255','0255'}: return '0255'
    return s.zfill(4) if s else ''

def result_to_core025_member(x):
    b=box_key(x)
    return b if b in MEMBERS else ''

def unique_straight_permutations(member):
    member=normalize_member(member)
    return sorted({''.join(p) for p in permutations(member,4)}) if member in MEMBERS else []

def load_upload(uploaded_file):
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    # Smart delimiter detection because many exported .txt files are actually comma-separated CSV.
    if name.endswith(".csv"):
        seps = [",", "\t", "|", ";"]
    elif name.endswith(".tsv"):
        seps = ["\t", ",", "|", ";"]
    else:
        seps = [",", "\t", "|", ";"]

    best_df = None
    best_cols = 0

    for sep in seps:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            if df.shape[1] > best_cols:
                best_df = df
                best_cols = df.shape[1]
            if "StreamKey" in df.columns:
                return df
        except Exception:
            continue

    if best_df is not None and best_cols > 1:
        return best_df

    try:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return pd.read_csv(io.BytesIO(raw), sep="\t", header=None, dtype=str, low_memory=False)


def normalize_history(raw):
    df=raw.copy(); df.columns=[str(c).strip() for c in df.columns]
    lower={c.lower():c for c in df.columns}
    if all(k in lower for k in ['date','state','game']):
        date_col,state_col,game_col=lower['date'],lower['state'],lower['game']
        result_col=None
        for cand in ['results','result','winning numbers','winning_numbers','winningnumbers']:
            if cand in lower: result_col=lower[cand]; break
        if result_col is None: result_col=df.columns[3]
        out=df[[date_col,state_col,game_col,result_col]].copy(); out.columns=['Date','State','Game','Results']
    else:
        if df.shape[1]<4: raise ValueError('History file must have Date, State, Game, Results.')
        out=df.iloc[:,:4].copy(); out.columns=['Date','State','Game','Results']
    out['Date']=pd.to_datetime(out['Date'], errors='coerce')
    out['Result']=out['Results'].apply(extract_pick4_digits)
    out=out[out['Date'].notna() & out['Result'].str.len().eq(4)].copy()
    out['StreamKey']=out['State'].astype(str).str.strip()+' | '+out['Game'].astype(str).str.strip()
    out['Core025Member']=out['Result'].apply(result_to_core025_member)
    return out.sort_values(['StreamKey','Date']).reset_index(drop=True)

def derive_seed(df,hist):
    h=hist.sort_values(['StreamKey','Date']).reset_index(drop=True)
    prior={}
    for stream,g in h.groupby('StreamKey'):
        g=g.sort_values('Date').reset_index(drop=True)
        for i in range(1,len(g)):
            prior[(str(stream), pd.Timestamp(g.loc[i,'Date']).normalize())]=str(g.loc[i-1,'Result']).zfill(4)
    vals=[]
    for _,r in df.iterrows():
        cur=extract_pick4_digits(r.get('seed',''))
        if len(cur)==4: vals.append(cur); continue
        key=(str(r.get('StreamKey','')), pd.Timestamp(r.get('Date')).normalize())
        vals.append(prior.get(key,''))
    return vals

def digits4(x):
    s=extract_pick4_digits(x)
    if len(s)!=4: s=re.sub(r'\D','',str(x or '')).zfill(4)[-4:]
    return [int(ch) for ch in s] if len(s)==4 else [0,0,0,0]

def repeat_shape(d):
    vals=sorted(pd.Series(d).value_counts().tolist(), reverse=True)
    return { (1,1,1,1):'all_unique',(2,1,1):'one_pair',(2,2):'two_pair',(3,1):'triple',(4,):'quad' }.get(tuple(vals),'other')

def structure_4(d):
    vals=sorted(pd.Series(d).value_counts().tolist(), reverse=True)
    return { (4,):'AAAA',(3,1):'AAAB',(2,2):'AABB',(2,1,1):'AABC' }.get(tuple(vals),'ABCD')

def sb(v): return 'sum_10_13' if v<=13 else 'sum_14_17' if v<=17 else 'sum_18_21' if v<=21 else 'sum_22_plus'
def spb(v): return 'spread_0_2' if v<=2 else 'spread_3_4' if v<=4 else 'spread_5_6' if v<=6 else 'spread_7_plus'
def consec(d):
    u=sorted(set(d)); return sum(1 for a,b in zip(u,u[1:]) if b-a==1)

def enrich(df):
    out=df.copy(); ds=out['seed'].apply(digits4); sums=[sum(d) for d in ds]; spreads=[max(d)-min(d) for d in ds]
    out['seed_sum']=sums; out['seed_spread']=spreads
    out['seed_even_cnt']=[sum(1 for x in d if x%2==0) for d in ds]
    out['seed_high_cnt']=[sum(1 for x in d if x>=5) for d in ds]
    out['seed_low_cnt']=[sum(1 for x in d if x<=4) for d in ds]
    out['seed_has9']=[int(9 in d) for d in ds]; out['seed_has0']=[int(0 in d) for d in ds]
    out['seed_parity_pattern']=[''.join('E' if x%2==0 else 'O' for x in d) for d in ds]
    out['seed_highlow_pattern']=[''.join('H' if x>=5 else 'L' for x in d) for d in ds]
    out['seed_repeat_shape']=[repeat_shape(d) for d in ds]
    for digit in range(10):
        out[f'cnt{digit}']=[d.count(digit) for d in ds]
        out[f'has{digit}']=[int(digit in d) for d in ds]
    out['even']=out['seed_even_cnt']; out['odd']=[4-int(v) for v in out['seed_even_cnt']]
    out['high']=out['seed_high_cnt']; out['low']=out['seed_low_cnt']
    out['parity_pattern']=out['seed_parity_pattern']; out['highlow_pattern']=out['seed_highlow_pattern']
    out['structure']=[structure_4(d) for d in ds]; out['unique']=[len(set(d)) for d in ds]
    out['max_rep']=[max(pd.Series(d).value_counts().tolist()) for d in ds]
    out['consec_links']=[consec(d) for d in ds]
    out['pair']=[int(max(pd.Series(d).value_counts().tolist())>=2) for d in ds]
    out['sum_bucket']=[sb(v) for v in sums]; out['spread_bucket']=[spb(v) for v in spreads]
    return out

def normalize_lab(raw,hist):
    df=raw.copy(); df.columns=[str(c).strip() for c in df.columns]
    if 'StreamKey' not in df.columns: raise ValueError('Lab/detail file missing StreamKey.')
    if 'Date' not in df.columns: raise ValueError('Lab/detail file missing Date.')
    if 'EventIndex' in df.columns and 'Mode' in df.columns:
        df['_pref']=df['Mode'].astype(str).eq('ALWAYS_TOP1__NONE').astype(int)
        df=df.sort_values(['EventIndex','_pref'], ascending=[True,False]).drop_duplicates('EventIndex').drop(columns=['_pref'])
    df['Date']=pd.to_datetime(df['Date'], errors='coerce')
    if 'Result' not in df.columns:
        if 'TrueStraight' in df.columns: df['Result']=df['TrueStraight']
        else: raise ValueError('Lab/detail file missing Result or TrueStraight.')
    df['Result']=df['Result'].apply(extract_pick4_digits)
    if 'seed' not in df.columns: df['seed']=''
    df['seed']=df['seed'].apply(lambda x: extract_pick4_digits(x) if extract_pick4_digits(x) else '')
    if df['seed'].str.len().ne(4).any(): df['seed']=derive_seed(df,hist)
    miss=df['seed'].astype(str).str.len().ne(4).sum()
    if miss: raise ValueError(f'Could not derive seed for {miss} rows. StreamKey/Date may not match history.')
    df['TrueMember']=df['TrueMember'].apply(normalize_member) if 'TrueMember' in df.columns else df['Result'].apply(result_to_core025_member)
    for c in ['PredictedMember','Top2_pred','ThirdMember']:
        df[c]=df[c].apply(normalize_member) if c in df.columns else ''
    for c in ['SingleRow','StreamRank','RowPercentile','StreamTier','PlayType','Top2Decision','Top2RiskScore','Top2ToTop1Ratio','Margin','RowVolatilityRate','RowTop1Rate','RowTop2Rate','RowTop3Rate','RowPlayType','RowPlayTypeReason','Top3Rescue','Top3RescueReasons']:
        if c not in df.columns: df[c]=''
    return enrich(df).reset_index(drop=True)

def dup_pattern(perm):
    s=str(perm).zfill(4); counts=pd.Series(list(s)).value_counts().to_dict(); dup=[d for d,c in counts.items() if c>1]
    if not dup: return 'none'
    d=sorted(dup)[0]; return '-'.join(str(i+1) for i,ch in enumerate(s) if ch==d)

def ordered_pairs(perm):
    s=str(perm).zfill(4); return [s[:2],s[1:3],s[2:]]

@st.cache_data(show_spinner=False)
def training_from_hist(hist_csv):
    hist=pd.read_csv(io.StringIO(hist_csv), dtype=str, low_memory=False); hist['Date']=pd.to_datetime(hist['Date'], errors='coerce')
    rows=[]
    for stream,g in hist.groupby('StreamKey'):
        g=g.sort_values('Date').reset_index(drop=True)
        for i in range(1,len(g)):
            member=result_to_core025_member(g.loc[i,'Result'])
            if member in MEMBERS:
                rows.append({'StreamKey':stream,'Date':g.loc[i,'Date'],'seed':g.loc[i-1,'Result'],'TrueMember':member,'StraightResult':extract_pick4_digits(g.loc[i,'Result']),'Result':extract_pick4_digits(g.loc[i,'Result'])})
    ev=pd.DataFrame(rows)
    if ev.empty: return ev
    ev=enrich(ev); ev['DupPattern']=ev['StraightResult'].apply(dup_pattern)
    ev['Pair12']=ev['StraightResult'].str[:2]; ev['Pair23']=ev['StraightResult'].str[1:3]; ev['Pair34']=ev['StraightResult'].str[2:]
    for i in range(4): ev[f'pos{i+1}']=ev['StraightResult'].str[i]
    return ev

def rate(df, mask, min_support=1):
    s=len(df)
    if s<min_support or s<=0: return 0.0,s
    return float(pd.Series(mask).sum())/s,s

def member_score(row, member):
    scores={}
    for m in MEMBERS:
        col=f'score_{m}'
        if col in row.index:
            try: scores[m]=float(row.get(col,0) or 0)
            except Exception: scores[m]=0.0
    if scores and max(scores.values())>0: return max(0,min(1,scores.get(member,0)/max(scores.values())))
    if member==normalize_member(row.get('PredictedMember','')): return 1.0
    if member==normalize_member(row.get('Top2_pred','')): return .65
    if member==normalize_member(row.get('ThirdMember','')): return .35
    return 0.0

def score_perm(row, member, perm, events):
    member=normalize_member(member); perm=str(perm).zfill(4); stream=str(row.get('StreamKey',''))
    me=events[events['TrueMember'].eq(member)]; se=me[me['StreamKey'].eq(stream)]
    te=me[(me['sum_bucket'].astype(str).eq(str(row.get('sum_bucket','')))) & (me['spread_bucket'].astype(str).eq(str(row.get('spread_bucket',''))))]
    she=me[(me['parity_pattern'].astype(str).eq(str(row.get('parity_pattern','')))) & (me['highlow_pattern'].astype(str).eq(str(row.get('highlow_pattern',''))))]
    ee=me[me['seed'].astype(str).eq(str(row.get('seed','')))]
    pos=[]; ps=[]
    for i,d in enumerate(perm):
        rg,sg=rate(me,me[f'pos{i+1}'].astype(str).eq(d),1); rs,ss=rate(se,se[f'pos{i+1}'].astype(str).eq(d),2); rt,st=rate(te,te[f'pos{i+1}'].astype(str).eq(d),3)
        pos.append(.55*rg+.30*rs+.15*rt); ps.append(sg+ss+st)
    position=sum(pos)/4 if pos else 0
    prs=[]; prs_s=[]
    for pair,col in zip(ordered_pairs(perm),['Pair12','Pair23','Pair34']):
        rg,sg=rate(me,me[col].astype(str).eq(pair),1); rs,ss=rate(se,se[col].astype(str).eq(pair),2); rh,sh=rate(she,she[col].astype(str).eq(pair),3)
        prs.append(.55*rg+.25*rs+.20*rh); prs_s.append(sg+ss+sh)
    pair_score=sum(prs)/3 if prs else 0
    exact,exs=rate(ee,ee['StraightResult'].astype(str).eq(perm),1); tr,ts=rate(te,te['StraightResult'].astype(str).eq(perm),3); shr,shs=rate(she,she['StraightResult'].astype(str).eq(perm),3)
    seedscore=.50*exact+.30*tr+.20*shr
    first,fs=rate(se,se['pos1'].astype(str).eq(perm[0]),2); last,ls=rate(se,se['pos4'].astype(str).eq(perm[3]),2)
    streamscore=(first+last)/2 if fs+ls>0 else 0
    recent=se.sort_values('Date').tail(20); rr,rs=rate(recent,recent['StraightResult'].astype(str).eq(perm),1); rf,rfs=rate(recent,recent['pos1'].astype(str).eq(perm[0]),2); rb,rbs=rate(recent,recent['pos4'].astype(str).eq(perm[3]),2)
    recentscore=.50*rr+.25*rf+.25*rb
    dp=dup_pattern(perm); dg,dgs=rate(me,me['DupPattern'].astype(str).eq(dp),1); ds,dss=rate(se,se['DupPattern'].astype(str).eq(dp),2); repeats=.70*dg+.30*ds
    mc=member_score(row,member); total=.25*mc+.25*position+.20*pair_score+.15*seedscore+.10*streamscore+.05*recentscore
    support=int(sum(ps)+sum(prs_s)+exs+ts+shs+fs+ls+rs+rfs+rbs+dgs+dss)
    return {'StraightPermutation':perm,'MemberConfidenceScore':round(mc*100,2),'PositionScore':round(position*100,2),'OrderedPairScore':round(pair_score*100,2),'SeedTransitionScore':round(seedscore*100,2),'StreamOrderScore':round(streamscore*100,2),'RecentOrderScore':round(recentscore*100,2),'RepeatPlacementScore':round(repeats*100,2),'StraightConfidenceScore':round(total*100,2),'EvidenceSupport':support,'DupPattern':dp,'StraightScoreFormula':'25% member + 25% position + 20% ordered_pair + 15% seed_transition + 10% stream_order + 5% recent_order'}

def generate(lab,hist,depth):
    events=training_from_hist(hist.to_csv(index=False))
    if events.empty: raise ValueError('No Core025 training events found in history.')
    ranked_rows=[]; summary=[]; prog=st.progress(0,text='Generating...'); total=len(lab)
    for i,(_,row) in enumerate(lab.iterrows(), start=1):
        true=extract_pick4_digits(row.get('Result','')); tm=normalize_member(row.get('TrueMember',result_to_core025_member(true)))
        parts=[]
        for member in MEMBERS:
            rows=[]
            for perm in unique_straight_permutations(member):
                rows.append({'EventIndex':i-1,'Date':row.get('Date',''),'StreamKey':row.get('StreamKey',''),'SingleRow':row.get('SingleRow',''),'StreamRank':row.get('StreamRank',''),'RowPercentile':row.get('RowPercentile',''),'seed':row.get('seed',''),'Result':true,'TrueMember':tm,'PredictedMember':normalize_member(row.get('PredictedMember','')),'Top2_pred':normalize_member(row.get('Top2_pred','')),'ThirdMember':normalize_member(row.get('ThirdMember','')),'PlayType':row.get('PlayType',''),'Top2Decision':row.get('Top2Decision',''),'Top2RiskScore':row.get('Top2RiskScore',''),'RowVolatilityRate':row.get('RowVolatilityRate',''),'RowPlayTypeReason':row.get('RowPlayTypeReason',''),'RankedMember':member,**score_perm(row,member,perm,events)})
            mdf=pd.DataFrame(rows).sort_values(['StraightConfidenceScore','EvidenceSupport'], ascending=[False,False]).reset_index(drop=True)
            mdf['MemberStraightRank']=mdf.index+1; mdf['MemberTop1Hit']=((mdf['StraightPermutation'].eq(true))&(mdf['MemberStraightRank']<=1)).astype(int); mdf['MemberTop3Hit']=((mdf['StraightPermutation'].eq(true))&(mdf['MemberStraightRank']<=3)).astype(int); mdf['MemberTop5Hit']=((mdf['StraightPermutation'].eq(true))&(mdf['MemberStraightRank']<=5)).astype(int)
            parts.append(mdf.copy()); ranked_rows.append(mdf[mdf['MemberStraightRank']<=int(depth)].copy())
        ev=pd.concat(parts,ignore_index=True)
        summ={'EventIndex':i-1,'Date':row.get('Date',''),'StreamKey':row.get('StreamKey',''),'SingleRow':row.get('SingleRow',''),'StreamRank':row.get('StreamRank',''),'RowPercentile':row.get('RowPercentile',''),'seed':row.get('seed',''),'Result':true,'TrueMember':tm,'PredictedMember':normalize_member(row.get('PredictedMember','')),'Top2_pred':normalize_member(row.get('Top2_pred','')),'ThirdMember':normalize_member(row.get('ThirdMember','')),'PlayType':row.get('PlayType',''),'Top2Decision':row.get('Top2Decision',''),'Top2RiskScore':row.get('Top2RiskScore',''),'RowVolatilityRate':row.get('RowVolatilityRate','')}
        for member in MEMBERS:
            sub=ev[ev['RankedMember'].eq(member)]; hit=sub[sub['StraightPermutation'].eq(true)]
            rank=int(hit['MemberStraightRank'].min()) if not hit.empty else None
            summ[f'{member}_TopStraight']=sub.iloc[0]['StraightPermutation'] if not sub.empty else ''; summ[f'{member}_TopScore']=sub.iloc[0]['StraightConfidenceScore'] if not sub.empty else ''; summ[f'{member}_HitRank']=rank; summ[f'{member}_Top1Hit']=int(rank is not None and rank<=1); summ[f'{member}_Top3Hit']=int(rank is not None and rank<=3); summ[f'{member}_Top5Hit']=int(rank is not None and rank<=5)
        for sel,member in [('AlwaysTop1',summ['PredictedMember']),('AlwaysTop2',summ['Top2_pred']),('AlwaysThird',summ['ThirdMember']),('TrueMemberOracle',tm)]:
            rank=summ.get(f'{member}_HitRank'); summ[f'{sel}_Member']=member; summ[f'{sel}_Top1Hit']=int(rank is not None and rank<=1); summ[f'{sel}_Top3Hit']=int(rank is not None and rank<=3); summ[f'{sel}_Top5Hit']=int(rank is not None and rank<=5)
        summary.append(summ)
        if i%10==0 or i==total: prog.progress(i/total,text=f'Generated {i}/{total}')
    prog.empty(); ranked=pd.concat(ranked_rows,ignore_index=True); summary=pd.DataFrame(summary)
    selectors=[]
    for sel in ['AlwaysTop1','AlwaysTop2','AlwaysThird','TrueMemberOracle']:
        h1=int(summary[f'{sel}_Top1Hit'].sum()); h3=int(summary[f'{sel}_Top3Hit'].sum()); h5=int(summary[f'{sel}_Top5Hit'].sum())
        selectors.append({'Selector':sel,'FullUniverseTotal':len(summary),'Top1Hits':h1,'Top3Hits':h3,'Top5Hits':h5,'Top5Pct':round(h5/len(summary)*100,2) if len(summary) else 0})
    return ranked, summary, pd.DataFrame(selectors)


def dual_download_buttons(label_base, df, base_filename, key_base):
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    txt_bytes = df.to_csv(index=False).encode('utf-8')
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            f'Download {label_base} CSV',
            csv_bytes,
            f'{base_filename}.csv',
            'text/csv',
            use_container_width=True,
            key=f'{key_base}_csv',
        )
    with c2:
        st.download_button(
            f'Download {label_base} TXT',
            txt_bytes,
            f'{base_filename}.txt',
            'text/plain',
            use_container_width=True,
            key=f'{key_base}_txt',
        )

st.set_page_config(page_title='Core025 All-3 Straight Generator', layout='wide')
st.title('Core025 All-3-Member Straight Rank Generator')
st.caption(BUILD_MARKER)
st.info('Single-file v6. Smart-loads TXT/CSV/TSV, including comma-separated .txt files, and produces CSV + TXT downloads.')
with st.sidebar:
    st.header('Inputs')
    history_file=st.file_uploader('Upload FULL HISTORY (.txt/.csv/.tsv)', type=['txt','csv','tsv'])
    lab_file=st.file_uploader('Upload LAB PER-EVENT or v78 DETAIL (.txt/.csv/.tsv)', type=['txt','csv','tsv'])
    depth=st.slider('Max ranked straights per member to export',5,24,12,1)
    run=st.button('GENERATE ALL-3-MEMBER FILES', type='primary', use_container_width=True)
if not history_file or not lab_file:
    st.warning('Upload both files.'); st.stop()
if not run:
    st.success('Files uploaded. Press the red GENERATE button.'); st.stop()
try:
    st.write('Loading files...')
    hist=normalize_history(load_upload(history_file)); lab=normalize_lab(load_upload(lab_file), hist)
    c1,c2,c3=st.columns(3); c1.metric('History rows',f'{len(hist):,}'); c2.metric('Events',f'{len(lab):,}'); c3.metric('Depth/member',depth)
    with st.spinner('Generating all-3-member straight ranks...'):
        ranked,summary,selector=generate(lab,hist,depth)
    st.success('Generation complete.')
    st.subheader('Selector Baseline Summary'); st.dataframe(selector, use_container_width=True, hide_index=True)
    st.subheader('Downloads')
    dual_download_buttons(
        'all_3_members_ranked_per_event__core025_v5',
        ranked,
        'all_3_members_ranked_per_event__core025_v5',
        'dl_ranked_v5',
    )
    dual_download_buttons(
        'all_3_members_event_summary__core025_v5',
        summary,
        'all_3_members_event_summary__core025_v5',
        'dl_summary_v5',
    )
    dual_download_buttons(
        'selector_baseline_summary__core025_v5',
        selector,
        'selector_baseline_summary__core025_v5',
        'dl_selector_v5',
    )
except Exception as e:
    st.error('Generator failed.'); st.exception(e)
