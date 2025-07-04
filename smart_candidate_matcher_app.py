# smart_candidate_matcher_app.py
import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- Configuration ---
st.set_page_config(page_title='Smart Candidate Matcher', layout='wide')

# Front-end에서 API 키 입력
api_key = st.sidebar.text_input('OpenAI API Key', type='password')
if not api_key:
    st.warning('OpenAI API Key를 입력해주세요.')
    st.stop()
openai.api_key = api_key

st.title('Smart Candidate Matcher')

# 이력서 파일 업로드 (TXT or CSV)
uploaded_files = st.file_uploader('이력서 업로드 (TXT 또는 CSV)', type=['txt', 'csv'], accept_multiple_files=True)
if uploaded_files:
    # CSV 파일 처리
    csv_files = [f for f in uploaded_files if f.name.lower().endswith('.csv')]
    if csv_files:
        if len(csv_files) > 1:
            st.warning('CSV 파일은 하나만 업로드 해주세요.')
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

    st.write('이력서 개수:', len(df))
    st.write('데이터 예시:', df.head())
    jd_text = st.text_area('직무기술서(JD) 텍스트 입력', height=150)

    if st.button('매칭 시작'):
        # JD 임베딩 생성 (v1+ OpenAI API 방식)
        jd_resp = openai.embeddings.create(
            input=jd_text,
            model='text-embedding-ada-002'
        )
        jd_embed = jd_resp.data[0].embedding

        scores, features_list = [], []
        for idx, row in df.iterrows():
            prompt = (
                f"다음 이력서를 분석하여 핵심 역량, 경험 키워드, 소프트 스킬 3가지를 JSON 형식으로 반환해줘:\n{row[resume_col]}"
            )
            resp = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[{'role': 'user', 'content': prompt}]
            )
            feat = resp.choices[0].message.content if resp.choices else '{}'

            emb_resp = openai.embeddings.create(
                input=row[resume_col],
                model='text-embedding-ada-002'
            )
            emb = emb_resp.data[0].embedding
            sim = cosine_similarity([jd_embed], [emb])[0][0]

            scores.append(sim)
            features_list.append(feat)

        df['유사도'] = scores
        df_sorted = df.sort_values(by='유사도', ascending=False)
        st.write('상위 지원자:', df_sorted[[resume_col, '유사도']].head())

        # 유사도 분포 히스토그램
        fig, ax = plt.subplots()
        ax.hist(df_sorted['유사도'], bins=20)
        ax.set_title('지원자 유사도 분포')
        st.pyplot(fig)

        # 워드 클라우드
        all_text = ' '.join(df_sorted[resume_col].astype(str).tolist())
        wc = WordCloud(width=800, height=400).generate(all_text)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(wc, interpolation='bilinear')
        ax2.axis('off')
        st.pyplot(fig2)
