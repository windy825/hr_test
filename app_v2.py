# HR 채용 적합도 분석기 (JD vs 자기소개서 분석)
# GPT 기반 분석 + 시각화 + 점수화 리포트 생성

import streamlit as st
import openai
import PyPDF2
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import json

st.set_page_config(page_title="채용 적합도 분석기", layout="wide")

# --- GPT API KEY 입력 ---
st.sidebar.title("🔐 GPT API Key")
api_key = st.sidebar.text_input("OpenAI API Key 입력", type="password")

if not api_key:
    st.warning("🔑 왼쪽 사이드바에 API Key를 입력해주세요.")
    st.stop()

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
        1. 핵심 경험과 키워드 (리스트 형태)
        2. 전반적 적합도 점수 (100점 만점 숫자)
        3. 강점 (리스트) / 우려사항 (리스트)
        4. 종합 의견 요약 (문단)
        5. 추천 여부 (강력 추천 / 가능 / 보통 / 비추천)
        6. 미래 잠재역량 또는 성장 가능성 (문장 2~3줄)
        결과는 JSON 형식으로 반환하고, 각 항목은 key로 명시해줘.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = response.choices[0].message.content
        except Exception as e:
            st.error("❌ GPT API 호출 중 오류 발생: " + str(e))
            st.stop()

    try:
        result_json = json.loads(result_text) if isinstance(result_text, str) else result_text
        st.success("✅ 분석 완료")

        # 적합도 점수 시각화
        score = result_json.get("전반적 적합도 점수", 0)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "전반적 적합도 점수", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4B9CD3"},
                'steps' : [
                    {'range': [0, 60], 'color': '#ffcccc'},
                    {'range': [60, 80], 'color': '#ffe066'},
                    {'range': [80, 100], 'color': '#b3ffb3'}],
            }))
        st.plotly_chart(fig, use_container_width=True)

        # 키워드 레이더 차트
        keywords = result_json.get("핵심 경험과 키워드", [])
        if keywords:
            df_kw = pd.DataFrame({"역량 키워드": keywords, "가중치": [1]*len(keywords)})
            st.markdown("### 🔍 JD 핵심 경험 및 키워드")
            st.dataframe(df_kw, use_container_width=True)

        # 강점/우려 radar chart
        strength = result_json.get("강점과 우려되는 점", {}).get("강점", [])
        weakness = result_json.get("강점과 우려되는 점", {}).get("우려되는 점", [])

        radar_labels = strength + weakness
        radar_scores = [8]*len(strength) + [3]*len(weakness)
        radar_df = pd.DataFrame(dict(역량=radar_labels, 점수=radar_scores))

        if not radar_df.empty:
            fig_radar = px.line_polar(radar_df, r='점수', theta='역량', line_close=True,
                                      color_discrete_sequence=['#636EFA'])
            st.markdown("### 📊 강점 vs 우려사항 분석")
            st.plotly_chart(fig_radar, use_container_width=True)

        # 종합 분석 리포트
        st.markdown("### 🧠 종합 분석 리포트")
        st.markdown(f"**📌 종합 요약:**\n\n{result_json.get('종합 의견 요약', '')}")
        st.markdown(f"**🌱 미래 잠재역량 진단:**\n\n{result_json.get('미래 잠재역량 또는 성장 가능성', '')}")
        st.markdown(f"**🏁 추천 여부:** ⭐️ {result_json.get('추천 여부', '')}")

    except Exception as e:
        st.error("❌ GPT 응답 파싱 중 오류가 발생했습니다.")
        st.text(result_text)
else:
    st.info("👈 왼쪽에서 자기소개서와 JD를 입력하고 실행을 눌러주세요!")
