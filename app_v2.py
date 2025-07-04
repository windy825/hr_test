# HR 채용 적합도 분석기 (JD vs 자기소개서 분석)
# GPT 기반 분석 + 시각화 + 점수화 리포트 생성 + PDF 저장 + 다중 지원자 비교 + 가중치 반영 + 이메일 전송

import streamlit as st
import openai
import PyPDF2
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import json
import base64
from io import BytesIO
from fpdf import FPDF
import smtplib
from email.message import EmailMessage

st.set_page_config(page_title="채용 적합도 분석기", layout="wide")

# --- GPT API KEY 입력 ---
st.sidebar.title("🔐 GPT API Key")
api_key = st.sidebar.text_input("OpenAI API Key 입력", type="password")

if not api_key:
    st.warning("🔑 왼쪽 사이드바에 API Key를 입력해주세요.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

# --- JD 및 가중치 입력 ---
st.sidebar.subheader("📌 JD 입력")
jd_input = st.sidebar.text_area("JD 또는 인사담당자 메모")

st.sidebar.subheader("⚖️ JD 중요도 가중치")
weights = {
    "핵심 경험과 키워드": st.sidebar.slider("경험 키워드 중요도", 1, 5, 3),
    "강점": st.sidebar.slider("강점 항목 중요도", 1, 5, 3),
    "우려사항": st.sidebar.slider("우려사항 민감도 (높을수록 감점)", 1, 5, 3),
    "미래 잠재역량": st.sidebar.slider("미래 잠재력 중요도", 1, 5, 2),
}

st.title("✨ GPT 기반 채용 적합도 분석기")

uploaded_files = st.file_uploader("📄 여러 명의 지원자 자기소개서 업로드 (PDF 또는 TXT)", type=["pdf", "txt"], accept_multiple_files=True)

email_enabled = st.checkbox("📧 리포트를 이메일로 발송하기")
email_address = st.text_input("수신 이메일 주소", value="") if email_enabled else None


def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])
    else:
        return file.read().decode("utf-8")

results = []

if st.button("📊 전체 지원자 적합도 분석 실행") and uploaded_files and jd_input:
    for file in uploaded_files:
        resume_text = extract_text(file)

        prompt = f"""
        JD 또는 기대사항: {jd_input}

        자기소개서:
        {resume_text}

        다음 항목에 대해 JSON 형식으로 분석해줘:
        {{
        "핵심 경험과 키워드": [...],
        "전반적 적합도 점수": 정수,
        "강점": [...],
        "우려사항": [...],
        "종합 의견 요약": "...",
        "추천 여부": "...",
        "미래 잠재역량 또는 성장 가능성": "..."
        }}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()
            json_start = content.find('{')
            json_data = content[json_start:]
            result = json.loads(json_data)
            result['파일명'] = file.name
            results.append(result)
        except Exception as e:
            st.error(f"❌ {file.name} 분석 중 오류 발생: {str(e)}")

    if results:
        st.success("✅ 전체 지원자 분석 완료")

        scores = []
        for r in results:
            raw_score = r.get("전반적 적합도 점수", 0)
            penalty = len(r.get("우려사항", [])) * weights["우려사항"]
            final_score = raw_score * weights["핵심 경험과 키워드"] + len(r.get("강점", [])) * weights["강점"] + weights["미래 잠재역량"] * 2 - penalty
            scores.append((r["파일명"], final_score))

        score_df = pd.DataFrame(scores, columns=["지원자", "가중 적합도 점수"])
        fig_all = px.bar(score_df, x="지원자", y="가중 적합도 점수", color="지원자",
                         title="📈 지원자별 가중 적합도 비교", text_auto=True)
        st.plotly_chart(fig_all, use_container_width=True)

        # 추가 시각화 - 적합도 Gauge + 항목별 분포
        for res in results:
            st.subheader(f"📋 {res['파일명']} 상세 분석")
            st.markdown(f"**✅ 적합도 점수:** {res['전반적 적합도 점수']}점 | **추천 여부:** {res['추천 여부']}")
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res['전반적 적합도 점수'],
                title={'text': "적합도 점수"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#4B9CD3"},
                    'steps': [
                        {'range': [0, 50], 'color': '#FFDDDD'},
                        {'range': [50, 75], 'color': '#FFE799'},
                        {'range': [75, 100], 'color': '#C4F4C4'}
                    ]
                }
            ))
            st.plotly_chart(gauge_fig, use_container_width=True)

            # 항목별 레이더 차트
            radar_labels = ['강점', '우려사항', '키워드 수', '잠재역량 점수(고정값)']
            radar_values = [len(res.get('강점', [])), len(res.get('우려사항', [])), len(res.get('핵심 경험과 키워드', [])), 3]
            radar_df = pd.DataFrame(dict(항목=radar_labels, 점수=radar_values))
            radar_fig = px.line_polar(radar_df, r='점수', theta='항목', line_close=True, title="📊 항목별 역량 분석")
            st.plotly_chart(radar_fig, use_container_width=True)

        # PDF 저장 및 이메일은 동일
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for res in results:
            lines = [
                f"지원자: {res['파일명']}",
                f"적합도 점수: {res.get('전반적 적합도 점수')}점",
                f"추천 여부: {res.get('추천 여부')}",
                "핵심 경험 및 키워드:", *res.get('핵심 경험과 키워드', []),
                "강점:", *res.get('강점', []),
                "우려사항:", *res.get('우려사항', []),
                f"미래 잠재역량: {res.get('미래 잠재역량 또는 성장 가능성')}",
                "종합 의견 요약:", res.get('종합 의견 요약'),
                "------------------------"
            ]
            for line in lines:
                pdf.cell(200, 10, txt=line, ln=True)

        pdf_output = BytesIO()
        pdf.output(pdf_output)
        b64_pdf = base64.b64encode(pdf_output.getvalue()).decode()
        href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="채용_분석_리포트.pdf">📥 PDF 리포트 다운로드</a>'
        st.markdown(href_pdf, unsafe_allow_html=True)

        if email_enabled and email_address:
            try:
                msg = EmailMessage()
                msg['Subject'] = '지원자 분석 리포트'
                msg['From'] = 'noreply@example.com'
                msg['To'] = email_address
                msg.set_content("채용 분석 리포트를 첨부드립니다.")
                msg.add_attachment(pdf_output.getvalue(), maintype='application', subtype='pdf', filename="채용_분석_리포트.pdf")

                with smtplib.SMTP('smtp.example.com', 587) as server:
                    server.starttls()
                    server.login('noreply@example.com', 'password')
                    server.send_message(msg)
                st.success("📧 이메일 발송 완료!")
            except Exception as e:
                st.error(f"이메일 발송 실패: {e}")
else:
    st.info("👈 JD 입력 및 지원자 파일 업로드 후 '적합도 분석 실행'을 눌러주세요")
