# smart_candidate_matcher_app.py
import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Configuration
oai_key = st.secrets['OPENAI_API_KEY']
openai.api_key = oai_key
st.set_page_config(page_title='Smart Candidate Matcher', layout='wide')

st.title('Smart Candidate Matcher')

uploaded = st.file_uploader('이력서와 JD 데이터 업로드 (CSV/Excel)', type=['csv', 'xlsx'])
if uploaded:
    df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
    resume_col = st.selectbox('이력서 텍스트 컬럼 선택', df.columns)
    jd_text = st.text_area('직무기술서(JD) 텍스트 입력', height=150)
    if st.button('매칭 시작'):
        jd_embed = openai.Embedding.create(input=jd_text, model='text-embedding-ada-002')['data'][0]['embedding']
        scores, features_list = [], []
        for _idx, row in df.iterrows():
            prompt = f"다음 이력서를 분석하여 핵심 역량, 경험 키워드, 소프트 스킬 3가지를 JSON 형식으로 반환해줘:\n{row[resume_col]}"
            resp = openai.ChatCompletion.create(model='gpt-4', messages=[{'role':'user','content':prompt}])
            feat = resp.choices[0].message.content if resp.choices else '{}'
            emb = openai.Embedding.create(input=row[resume_col], model='text-embedding-ada-002')['data'][0]['embedding']
            sim = cosine_similarity([jd_embed], [emb])[0][0]
            scores.append(sim)
            features_list.append(feat)
        df['유사도'] = scores
        df_sorted = df.sort_values(by='유사도', ascending=False)
        st.write('상위 지원자:', df_sorted[[resume_col, '유사도']].head())
        # 시각화
        fig, ax = plt.subplots(); ax.hist(df_sorted['유사도'], bins=20); ax.set_title('유사도 분포'); st.pyplot(fig)
        all_text = ' '.join(df_sorted[resume_col].astype(str)); wc = WordCloud(width=800, height=400).generate(all_text)
        fig2, ax2 = plt.subplots(figsize=(10,5)); ax2.imshow(wc, interpolation='bilinear'); ax2.axis('off'); st.pyplot(fig2)
