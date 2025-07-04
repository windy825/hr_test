# HR 5대 직무별 GPT 기반 Streamlit 데모 구현 코드 템플릿
# 고급 기능 포함: PDF 저장, WordCloud, 감정분석, CSV/PDF 업로드, API Key 입력

import streamlit as st
import openai
import os
import pdfkit
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import pandas as pd
import PyPDF2

# GPT API Key 입력 (일회용)
st.sidebar.title("🔐 GPT API Key 입력")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openai.api_key = api_key

####################
# 1. 채용: JD 기반 이력서 평가기
####################
def resume_evaluator():
    st.header("1. 채용 - JD 기반 이력서 평가기")
    jd = st.text_area("📌 채용공고 (JD)를 입력하세요:")
    resume = st.file_uploader("📄 이력서를 업로드하세요 (txt/pdf)", type=["txt", "pdf"])
    resume_text = ""

    if resume:
        if resume.type == "application/pdf":
            reader = PyPDF2.PdfReader(resume)
            for page in reader.pages:
                resume_text += page.extract_text()
        else:
            resume_text = resume.read().decode("utf-8")

    if st.button("이력서 평가하기") and jd and resume_text:
        prompt = f"""
        다음 채용공고(JD)에 적합한지 이력서를 평가해주세요. 핵심 역량, 경력 연관성, 기술 스킬을 기준으로 점수화하고, 인터뷰 질문 3개도 생성해주세요.
        JD:
        {jd}

        이력서:
        {resume_text}
        """
        response = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
        result = response["choices"][0]["message"]["content"]
        st.markdown(result)
        if st.download_button("📥 평가결과 PDF 다운로드", data=result, file_name="resume_result.pdf"):
            pdfkit.from_string(result, "resume_result.pdf")

####################
# 2. 교육: 맞춤 학습 경로 추천기
####################
def learning_recommender():
    st.header("2. 교육 - 직무 기반 학습 로드맵 추천기")
    job = st.text_input("🔧 직무명 입력 (예: 마케팅 매니저)")
    level = st.selectbox("📊 경험 수준 선택", ["주니어", "미들", "시니어"])
    focus = st.text_input("🎯 중점 역량/주제 (예: 데이터 기반 의사결정)")

    if st.button("학습 로드맵 추천"):
        prompt = f"직무: {job}, 수준: {level}, 집중역량: {focus}에 맞춘 학습 경로를 단계별로 설계해줘"
        response = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
        st.markdown(response["choices"][0]["message"]["content"])

####################
# 3. 평가: 피드백 문장 생성기
####################
def performance_feedback():
    st.header("3. 평가 - 피드백 자동 생성기")
    trait = st.multiselect("평가 항목 선택", ["리더십", "책임감", "커뮤니케이션", "협업", "문제해결"])
    example = st.text_area("📌 피평가자 관련 사례 입력")

    if st.button("피드백 문장 생성"):
        prompt = f"항목: {', '.join(trait)}\n사례: {example}\n공감 피드백 작성"
        response = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
        st.markdown(response["choices"][0]["message"]["content"])

####################
# 4. 보상: 보상 제안 생성기
####################
def compensation_planner():
    st.header("4. 보상 - 직무 기반 보상 제안 리포트")
    role = st.text_input("직무명 입력")
    exp = st.slider("경력 연차", 0, 30, 3)
    region = st.selectbox("근무 지역", ["서울", "수도권", "지방", "원격"])

    if st.button("보상 제안 생성"):
        prompt = f"직무: {role}, 경력: {exp}, 지역: {region}에 적절한 보상안 제안"
        response = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
        st.markdown(response["choices"][0]["message"]["content"])

####################
# 5. 조직문화: 설문 요약 + 감정분석 + 워드클라우드
####################
def culture_survey_analyzer():
    st.header("5. 조직문화 - 설문 분석기")
    survey = st.text_area("📄 사내 설문/의견 텍스트 입력")
    analysis_type = st.radio("분석 유형", ["요약", "감정분석", "워드클라우드"])

    if st.button("설문 분석 실행"):
        if analysis_type == "요약":
            prompt = f"다음 내용을 요약해줘:\n{survey}"
            response = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
            st.markdown(response["choices"][0]["message"]["content"])
        elif analysis_type == "감정분석":
            blob = TextBlob(survey)
            st.write(f"긍정도 점수: {blob.sentiment.polarity:.2f}")
        elif analysis_type == "워드클라우드":
            wc = WordCloud(font_path=None, width=600, height=400).generate(survey)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

####################
# 6. CSV 기반 GPT 분석 (예: 교육 요청서)
####################
def csv_analyzer():
    st.header("6. CSV 기반 분석기 (예: 교육 요청서)")
    csv_file = st.file_uploader("CSV 업로드", type="csv")
    if csv_file:
        df = pd.read_csv(csv_file)
        st.write("업로드된 데이터:", df.head())
        if st.button("GPT 요약 생성"):
            prompt = f"다음 데이터로 인사이트 요약해줘:\n{df.to_string()}"
            response = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
            st.markdown(response["choices"][0]["message"]["content"])

####################
# 메인 실행
####################
def main():
    st.set_page_config(page_title="HR AI Toolkit", layout="wide")
    st.title("💼 HR AI Toolkit - GPT 기반 인사 자동화 도구")

    menu = st.sidebar.radio("기능 선택", ["1. 채용", "2. 교육", "3. 평가", "4. 보상", "5. 조직문화", "6. CSV 분석"])
    if menu == "1. 채용":
        resume_evaluator()
    elif menu == "2. 교육":
        learning_recommender()
    elif menu == "3. 평가":
        performance_feedback()
    elif menu == "4. 보상":
        compensation_planner()
    elif menu == "5. 조직문화":
        culture_survey_analyzer()
    elif menu == "6. CSV 분석":
        csv_analyzer()

if __name__ == "__main__":
    main()
