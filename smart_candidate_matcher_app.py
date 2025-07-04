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

# 사이드바
st.sidebar.title('설정')
api_key = st.sidebar.text_input('OpenAI API Key 입력', type='password')
file_types = st.sidebar.multiselect('이력서 파일 형식', ['TXT', 'CSV'], default=['TXT', 'CSV'])

if not api_key:
    st.sidebar.warning('API Key를 입력해주세요.')
    st.stop()
client = OpenAI(api_key=api_key)

# 메인 UI
st.title('스마트 후보 매칭 대시보드')

uploaded = st.file_uploader(
    '이력서 업로드 (TXT 또는 CSV, 다중 선택 가능)',
    type=[f.lower() for f in file_types], accept_multiple_files=True
)

# 데이터 로드
if uploaded:
    resumes = []
    for f in uploaded:
        if f.name.lower().endswith('.csv'):
            df_csv = pd.read_csv(f)
            text_col = st.selectbox(f'CSV 컬럼 선택: {f.name}', df_csv.columns)
            for _, r in df_csv.iterrows():
                resumes.append({'text': r[text_col], 'source': f.name})
        else:
            text = f.getvalue().decode('utf-8', errors='ignore')
            resumes.append({'text': text, 'source': f.name})
    df = pd.DataFrame(resumes)

    # 요약 정보
    total = len(df)
    char_counts = df['text'].str.len()
    word_counts = df['text'].str.split().str.len()
    avg_sim = 0  # 초기값

    # JD 입력
    jd = st.text_area('직무기술서(JD) 입력', height=150)
    if st.button('분석 시작'):
        # 임베딩
        jd_emb = client.embeddings.create(input=jd, model='text-embedding-ada-002').data[0].embedding
        sims, features = [], []
        for txt in df['text']:
            # 유사도 계산
            emb = client.embeddings.create(input=txt, model='text-embedding-ada-002').data[0].embedding
            sims.append(cosine_similarity([jd_emb], [emb])[0][0])
            # 핵심 정보 추출
            resp = client.chat.completions.create(
                model='gpt-4',
                messages=[{'role':'user', 'content': f'이 이력서를 분석하여 핵심 역량, 경험 키워드, 소프트 스킬 3가지를 JSON으로 주세요:\n{txt}'}]
            )
            try:
                features.append(json.loads(resp.choices[0].message.content))
            except:
                features.append({})
        df['유사도'] = sims
        df['문자수'] = char_counts
        df['단어수'] = word_counts
        avg_sim = np.round(np.mean(sims), 3)

        # 상단 KPI
        k1, k2, k3 = st.columns(3)
        k1.metric('지원자 수', total)
        k2.metric('평균 유사도', avg_sim)
        k3.metric('최고 유사도', np.round(np.max(sims), 3))

        # 상위 5명 데이터
        top5 = df.sort_values('유사도', ascending=False).head(5)
        st.subheader('상위 5명 지원자')
        st.table(top5[['source', '유사도', '문자수', '단어수']])

        # 시각화 레이아웃
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('유사도 분포')
            fig, ax = plt.subplots()
            ax.hist(sims, bins=15)
            ax.set_xlabel('유사도')
            ax.set_ylabel('지원자 수')
            st.pyplot(fig)
        with col2:
            st.subheader('문자수 vs 유사도')
            fig2, ax2 = plt.subplots()
            ax2.scatter(df['문자수'], df['유사도'])
            ax2.set_xlabel('문자수')
            ax2.set_ylabel('유사도')
            st.pyplot(fig2)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader('단어수 vs 유사도')
            fig3, ax3 = plt.subplots()
            ax3.scatter(df['단어수'], df['유사도'])
            ax3.set_xlabel('단어수')
            ax3.set_ylabel('유사도')
            st.pyplot(fig3)
        with col4:
            st.subheader('상위 5명 유사도')
            fig4, ax4 = plt.subplots()
            ax4.barh([f'{i+1}번' for i in range(len(top5))], top5['유사도'])
            ax4.invert_yaxis()
            ax4.set_xlabel('유사도')
            st.pyplot(fig4)

        # 상세 프로필
        st.subheader('상위 1명 분석')
        prof = features[np.argmax(sims)]
        st.json(prof)
