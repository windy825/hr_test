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

# Custom CSS for styling
st.markdown("""
<style>
/* 배경 그라데이션 */
body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
/* 카드 스타일 */
.card { background: rgba(255,255,255,0.9); border-radius: 12px; padding: 16px; margin-bottom: 16px; }
.kpi { text-align: center; padding: 12px; border-radius: 8px; background: #ffffff; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.upload-area { border: 2px dashed #cbd5e0; border-radius: 8px; padding: 32px; text-align: center; color: #718096; transition: border-color .3s; }
.upload-area:hover { border-color: #4299e1; background: rgba(235,248,255,0.3); }
.progress-bar { height: 10px; border-radius: 5px; background: #4299e1; }
</style>
""", unsafe_allow_html=True)

# Sidebar: Settings
st.sidebar.title('설정')
api_key = st.sidebar.text_input('OpenAI API Key 입력', type='password')
file_types = st.sidebar.multiselect('이력서 파일 형식', ['TXT','CSV'], default=['TXT','CSV'])

if not api_key:
    st.sidebar.warning('API Key를 입력해주세요.')
    st.stop()
client = OpenAI(api_key=api_key)

# Main Header
st.markdown("""
<div class='card'>
  <h1 style='margin:0; font-size:32px; color:#2d3748;'><i class='fas fa-user-tie' style='color:#667eea;'></i>&nbsp;스마트 후보 매칭 대시보드</h1>
  <p style='color:#4a5568; margin-top:4px;'>GPT 기반 AI 매칭 및 다채로운 시각화 분석</p>
</div>
""", unsafe_allow_html=True)

# Upload Area
uploaded = st.file_uploader(
    '이력서 업로드 (TXT/CSV, 다중 선택 가능)',
    type=[f.lower() for f in file_types], accept_multiple_files=True
)

if uploaded:
    st.markdown("<div class='card'><strong>⏳ 업로드 및 분석 준비 중...</strong><div class='upload-area'>파일을 여기에 드래그하거나 클릭해주세요</div></div>", unsafe_allow_html=True)
    resumes=[]
    for f in uploaded:
        if f.name.lower().endswith('.csv'):
            df_csv=pd.read_csv(f)
            col=st.selectbox(f'CSV 컬럼 선택: {f.name}', df_csv.columns)
            for _,r in df_csv.iterrows(): resumes.append({'text':r[col],'source':f.name})
        else:
            text=f.getvalue().decode('utf-8', errors='ignore')
            resumes.append({'text':text,'source':f.name})
    df=pd.DataFrame(resumes)

    # Input JD
    jd=st.text_area('직무기술서(JD) 입력', height=150)
    run=st.button('🚀 분석 시작')

    if run:
        # Progress indicator
        progress=st.progress(0)
        steps=['JD 임베딩','유사도 계산','핵심정보 추출','시각화 준비']
        for i,step in enumerate(steps):
            st.info(f"{step} 진행 중...")
            progress.progress((i+1)/len(steps))

        # Compute embeddings & features
        jd_emb=client.embeddings.create(input=jd,model='text-embedding-ada-002').data[0].embedding
        sims=[]; feats=[]
        df['문자수']=df['text'].str.len()
        df['단어수']=df['text'].str.split().str.len()
        for idx,row in df.iterrows():
            emb=client.embeddings.create(input=row['text'],model='text-embedding-ada-002').data[0].embedding
            sims.append(cosine_similarity([jd_emb],[emb])[0][0])
            chat=client.chat.completions.create(
                model='gpt-4',
                messages=[{'role':'user','content':f"이 이력서를 JSON으로 분석해줘:\n{row['text']}"}]
            )
            try: feats.append(json.loads(chat.choices[0].message.content))
            except: feats.append({})
        df['유사도']=np.round(sims,3)
        avg_sim=np.round(df['유사도'].mean(),3)
        max_sim=np.round(df['유사도'].max(),3)

        # KPI Metrics
        k1,k2,k3=st.columns(3)
        k1.markdown(f"<div class='kpi'><h3>지원자 수</h3><p><strong>{len(df)}</strong></p></div>",unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi'><h3>평균 유사도</h3><p><strong>{avg_sim}</strong></p></div>",unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi'><h3>최고 유사도</h3><p><strong>{max_sim}</strong></p></div>",unsafe_allow_html=True)

        # Tabs for detailed views
        tab1,tab2,tab3=st.tabs(['📊 개요','📈 시각화','🔍 상세'])

        with tab1:
            st.header('지원자 요약')
            st.table(df[['source','문자수','단어수','유사도']].sort_values('유사도',ascending=False).head(10))

        with tab2:
            # Histogram
            st.subheader('유사도 분포')
            fig,ax=plt.subplots();ax.hist(df['유사도'],bins=20);ax.set_title('유사도 히스토그램');st.pyplot(fig)
            # Scatter
            st.subheader('문자수 vs 유사도')
            fig2,ax2=plt.subplots();ax2.scatter(df['문자수'],df['유사도']);st.pyplot(fig2)
            # Bar top5
            st.subheader('상위 5명 유사도 비교')
            top5=df.nlargest(5,'유사도')
            fig3,ax3=plt.subplots();ax3.barh(top5['source'],top5['유사도']);ax3.invert_yaxis();st.pyplot(fig3)

        with tab3:
            st.header('상위 1명 분석')
            prof=feats[np.argmax(sims)]
            st.json(prof)
