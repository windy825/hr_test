# HR 채용 적합도 분석기 (JD vs 자기소개서 분석)
# GPT 기반 분석 + 시각화 + 점수화 리포트 생성

import streamlit as st
import openai
import PyPDF2
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import re
import textwrap

st.set_page_config(page_title="채용 적합도 분석기", layout="wide")

# --- GPT API KEY 입력 ---
st.sidebar.title("🔐 GPT API Key")
api_key = st.sidebar.text_input("OpenAI API Key 입력", type="password")
client = openai.OpenAI(api_key=api_key)

# --- 앱 UI 구성 ---
st.markdown("""
    <h1 style='color:#4B9CD3;'>✨ GPT 기반 채용 적합도 분석기</h1>
    <h4 style='color:gray;'>자기소개서 + JD 입력 → GPT가 자동 분석 + 점수화 + 시각화</h4>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 지원자 자기소개서 업로드")
    resume_file = st.file_uploader("PDF 또는 텍스트 파일 업로드", type=["pdf", "txt"])
    resume_text = ""
    if resume_file:
        if resume_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(resume_file)
            for page in reader.pages:
                resume_text += page.extract_text()
        else:
            resume_text = resume_file.read().decode("utf-8")

with col2:
    st.subheader("🧾 JD 또는 인사담당자 메모 입력")
    jd_input = st.text_area("지원자에게 기대하는 내용이나 JD를 입력하세요")

if st.button("📊 적합도 분석 실행") and resume_text and jd_input:
    with st.spinner("GPT가 분석 중입니다..."):
        prompt = f"""
        아래는 한 명의 지원자의 자기소개서이며, 아래 JD에 얼마나 적합한 인재인지 분석해줘. 
        JD 또는 기대사항: {jd_input}

        자기소개서:
        {resume_text}

        다음 항목에 대해 분석해줘:
        1. JD에 부합하는 핵심 경험과 키워드
        2. 전반적 적합도 점수 (100점 만점)
        3. 강점과 우려되는 점
        4. 종합 의견 요약
        5. 추천 여부 (강력 추천 / 가능 / 보통 / 비추천)
        결과는 JSON 형식으로 항목별 출력해줘.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.choices[0].message.content

    try:
        result_json = eval(result_text)
        st.success("✅ 분석 완료")

        # 점수 시각화
        score = result_json.get("2. 전반적 적합도 점수 (100점 만점)", 0)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "적합도 점수", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4B9CD3"},
                'steps' : [
                    {'range': [0, 60], 'color': '#ffcccc'},
                    {'range': [60, 80], 'color': '#ffe066'},
                    {'range': [80, 100], 'color': '#b3ffb3'}],
            }))
        st.plotly_chart(fig, use_container_width=True)

        # 주요 분석 결과
        st.markdown("### 🧠 분석 요약")
        st.markdown(f"**1. JD 키워드/경험 적합성:**\n\n{result_json.get('1. JD에 부합하는 핵심 경험과 키워드', '')}")
        st.markdown(f"**3. 강점과 우려되는 점:**\n\n{result_json.get('3. 강점과 우려되는 점', '')}")
        st.markdown(f"**4. 종합 의견 요약:**\n\n{result_json.get('4. 종합 의견 요약', '')}")
        st.markdown(f"**5. 추천 여부:** ⭐️ {result_json.get('5. 추천 여부 (강력 추천 / 가능 / 보통 / 비추천)', '')}")

    except Exception as e:
        st.error("❌ GPT 응답 파싱 중 오류가 발생했습니다.")
        st.text(result_text)
else:
    st.info("👈 왼쪽에서 자기소개서와 JD를 입력하고 실행을 눌러주세요!")
