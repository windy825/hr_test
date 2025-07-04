# smart_candidate_matcher_app.py
import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import json

# --- Configuration ---
st.set_page_config(page_title='Smart Candidate Matcher', layout='wide')

# Front-end에서 API 키 입력
api_key = st.sidebar.text_input('OpenAI API Key 입력', type='password')
if not api_key:
    st.warning('OpenAI API Key를 입력해주세요.')
    st.stop()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

st.title('스마트 후보 매칭 대시보드')

# 이력서 파일 업로드 (TXT 또는 CSV)
uploaded_files = st.file_uploader(
    '이력서 업로드 (TXT 또는 CSV, 다중 선택 가능)',
    type=['txt', 'csv'], accept_multiple_files=True
)
if uploaded_files:
    # CSV 파일 처리
    csv_files = [f for f in uploaded_files if f.name.lower().endswith('.csv')]
    if csv_files:
        if len(csv_files) > 1:
            st.warning('CSV 파일은 하나만 업로드해주세요.')
            st.stop()
        df = pd.read_csv(csv_files[0])
        resume_col = st.selectbox('이력서 텍스트 컬럼 선택', df.columns)
    else:
        # TXT 파일 여러 개 처리
        resumes = []
        for f in uploaded_files:
            text = f.getvalue().decode('utf-8', errors='ignore')
            resumes.append({'resume_text': text, 'filename': f.name})
        df = pd.DataFrame(resumes)
        resume_col = 'resume_text'

    st.write(f'업로드된 이력서 수: {len(df)}')
    jd_text = st.text_area('직무기술서(JD) 입력', height=150)

    if st.button('매칭 실행'):
        # JD 임베딩
        jd_resp = client.embeddings.create(input=jd_text, model='text-embedding-ada-002')
        jd_embed = jd_resp.data[0].embedding

        # 이력서별 분석 및 유사도
        df['문자수'] = df[resume_col].apply(len)
        df['단어수'] = df[resume_col].apply(lambda x: len(x.split()))
        scores = []
        features = []
        for idx, row in df.iterrows():
            text = row[resume_col]
            # GPT 분석
            prompt = (
                f"이 이력서를 분석하여 핵심 역량, 경험 키워드, 소프트 스킬 3가지를 JSON 형식으로 반환해주세요:\n{text}"
            )
            chat = client.chat.completions.create(
                model='gpt-4',
                messages=[{'role': 'user', 'content': prompt}]
            )
            try:
                feats = json.loads(chat.choices[0].message.content)
            except:
                feats = {}
            features.append(feats)

            # 임베딩 및 유사도
            emb_resp = client.embeddings.create(input=text, model='text-embedding-ada-002')
            emb = emb_resp.data[0].embedding
            sim = cosine_similarity([jd_embed], [emb])[0][0]
            scores.append(sim)

        df['유사도'] = scores
        df_sorted = df.sort_values(by='유사도', ascending=False)
        top5 = df_sorted.head(5)

        # 상위 지원자 표
        st.subheader('상위 5명 지원자')
        st.dataframe(top5[[resume_col, '유사도', '문자수', '단어수']])

        # 유사도 분포 히스토그램
        st.subheader('유사도 분포')
        fig, ax = plt.subplots()
        ax.hist(df['유사도'], bins=20)
        ax.set_xlabel('유사도')
        ax.set_ylabel('지원자 수')
        st.pyplot(fig)

        # 문자수 vs 유사도 산점도
        st.subheader('문자수 vs 유사도')
        fig2, ax2 = plt.subplots()
        ax2.scatter(df['문자수'], df['유사도'])
        ax2.set_xlabel('문자수')
        ax2.set_ylabel('유사도')
        st.pyplot(fig2)

        # 단어수 vs 유사도 산점도
        st.subheader('단어수 vs 유사도')
        fig3, ax3 = plt.subplots()
        ax3.scatter(df['단어수'], df['유사도'])
        ax3.set_xlabel('단어수')
        ax3.set_ylabel('유사도')
        st.pyplot(fig3)

        # 상위 5명 유사도 막대그래프
        st.subheader('상위 5명 유사도 비교')
        fig4, ax4 = plt.subplots()
        ax4.barh(
            [f"지원자 {i+1}" for i in range(len(top5))],
            top5['유사도']
        )
        ax4.set_xlabel('유사도')
        ax4.invert_yaxis()
        st.pyplot(fig4)

        # 레이더 차트: 유사도, 문자수, 단어수
        st.subheader('레이더 차트 비교')
        categories = ['유사도', '문자수', '단어수']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        fig5, ax5 = plt.subplots(subplot_kw={'polar': True})
        for i, (_, row) in enumerate(top5.iterrows()):
            vals = [row['유사도'], row['문자수'] / df['문자수'].max(), row['단어수'] / df['단어수'].max()]
            vals += vals[:1]
            ax5.plot(angles, vals, label=f'지원자 {i+1}')
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig5)
