# smart_candidate_matcher_app.py
import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import json

# --- Configuration ---
st.set_page_config(page_title='스마트 후보 매칭 대시보드', layout='wide')

# Load FontAwesome
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" integrity="sha512-pT+Zw9sduHcR+e4Vf+3VxYj6IK9k5sH2+6B+LeD2IyN1S5r8hGm0b0p+1OQEgY3YbTcDsIqY5QJn5bZh9ep2PQ==" crossorigin="anonymous" referrerpolicy="no-referrer" />
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
body { background: linear-gradient(135deg, #f0f4ff 0%, #e2e8f0 100%); }
.glass { background: rgba(255,255,255,0.8); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
.kpi-card { border-radius: 8px; color: white; padding: 16px; position: relative; overflow: hidden; }
.kpi-title { font-size: 14px; }
.kpi-value { font-size: 28px; font-weight: bold; margin-top: 8px; }
.kpi-sub { font-size: 12px; opacity: 0.8; }
.progress-bar { height: 6px; border-radius: 3px; background: rgba(255,255,255,0.5); margin-top: 10px; }
.progress-bar-fill { height: 6px; border-radius: 3px; background: white; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("설정")
api_key = st.sidebar.text_input('🔑 OpenAI API Key', type='password')
if not api_key:
    st.sidebar.warning('API Key를 입력해주세요.')
    st.stop()
client = OpenAI(api_key=api_key)

# Main UI
st.markdown('<div class="glass"><h1 style="font-size:32px; margin:0;"><i class="fas fa-user-tie" style="color:#4f46e5;"></i> 스마트 후보 매칭 대시보드</h1><p style="margin:0; opacity:0.7;">AI 기반 통합 지원자 분석 및 매칭</p></div>', unsafe_allow_html=True)

# Upload section
st.markdown('<div class="glass"><h2>1. 이력서 업로드 및 JD 입력</h2></div>', unsafe_allow_html=True)
files = st.file_uploader('이력서를 TXT 또는 CSV로 업로드하세요 (다중 선택 가능)', type=['txt','csv'], accept_multiple_files=True)
jd = st.text_area('직무기술서(JD)를 입력하세요', height=150)
run = st.button('🚀 분석 시작')

if files and jd and run:
    # Read resumes
    resumes = []
    for f in files:
        if f.name.lower().endswith('.csv'):
            df_csv = pd.read_csv(f)
            col = st.selectbox(f'CSV 컬럼 선택: {f.name}', df_csv.columns)
            for _,r in df_csv.iterrows(): resumes.append({'text':r[col],'src':f.name})
        else:
            text = f.getvalue().decode('utf-8', errors='ignore')
            resumes.append({'text':text,'src':f.name})
    df = pd.DataFrame(resumes)
    n = len(df)
    df['chars'] = df['text'].str.len()
    df['words'] = df['text'].str.split().str.len()

    # KPI preliminary
    jd_emb = client.embeddings.create(input=jd, model='text-embedding-ada-002').data[0].embedding
    sims = []
    feats = []
    progress = st.progress(0)
    for i,row in df.iterrows():
        emb = client.embeddings.create(input=row['text'], model='text-embedding-ada-002').data[0].embedding
        sims.append(cosine_similarity([jd_emb],[emb])[0][0])
        resp = client.chat.completions.create(model='gpt-4', messages=[{'role':'user','content':f"이 이력서의 핵심 역량, 경험 키워드, 소프트 스킬 3가지 JSON으로 요약:
{row['text']}"}])
        try: feats.append(json.loads(resp.choices[0].message.content))
        except: feats.append({})
        progress.progress((i+1)/n)
    df['sim'] = np.round(sims,3)
    avg = df['sim'].mean(); maxv = df['sim'].max(); pct = (df['sim']>0.8).mean()*100

    # KPI display
    colors = [('#f87171','#b91c1c'), ('#4ade80','#166534'), ('#93c5fd','#1e40af')]
    titles = ['평균 유사도', '최고 유사도', '80% 이상 비율']
    values = [f"{avg:.2f}", f"{maxv:.2f}", f"{pct:.1f}%"]
    kcols = st.columns(3)
    for idx, colc in enumerate(kcols):
        bg1,bg2 = colors[idx]
        colc.markdown(f"<div class='kpi-card' style='background:linear-gradient(45deg,{bg1}, {bg2});'><div class='kpi-title'>{titles[idx]}</div><div class='kpi-value'>{values[idx]}</div><div class='progress-bar'><div class='progress-bar-fill' style='width:{values[idx]};'></div></div></div>", unsafe_allow_html=True)

    # Tabs
    tab1,tab2,tab3 = st.tabs(['지원자 개요','시각화','상위 1명 프로필'])
    with tab1:
        st.dataframe(df[['src','chars','words','sim']].sort_values('sim', ascending=False).head(10))
    with tab2:
        st.subheader('유사도 분포')
        fig1,ax1 = plt.subplots(); ax1.hist(df['sim'], bins=20, color='#4f46e5'); st.pyplot(fig1)
        st.subheader('문자수 vs 유사도')
        fig2,ax2 = plt.subplots(); ax2.scatter(df['chars'], df['sim'], c='#fbbf24'); st.pyplot(fig2)
        st.subheader('상위 5명 비교')
        top5 = df.nlargest(5,'sim')
        fig3,ax3 = plt.subplots(); ax3.barh(top5.index, top5['sim'], color='#10b981'); ax3.invert_yaxis(); st.pyplot(fig3)
    with tab3:
        st.json(feats[np.argmax(sims)])
