import streamlit as st
import openai
import PyPDF2
import pandas as pd
import plotly.express as px
import json
from io import BytesIO
from email.message import EmailMessage
from weasyprint import HTML
import smtplib

# --- 기본 설정 ---
st.set_page_config(page_title="채용 적합도 분석기", layout="wide")
st.title("✨ GPT 기반 채용 적합도 분석기")

# --- API 키 입력 ---
st.sidebar.title("🔐 GPT API Key")
api_key = st.sidebar.text_input("OpenAI API Key 입력", type="password")
if not api_key:
    st.warning("🔑 API Key를 입력해주세요.")
    st.stop()
client = openai.OpenAI(api_key=api_key)

# --- JD 및 가중치 입력 ---
st.sidebar.subheader("📌 JD 입력")
jd_input = st.sidebar.text_area("JD 또는 인사담당자 메모")

st.sidebar.subheader("⚖️ JD 중요도 가중치")
weights = {
    "핵심 경험과 키워드": st.sidebar.slider("경험 키워드 중요도", 1, 5, 3),
    "강점": st.sidebar.slider("강점 항목 중요도", 1, 5, 3),
    "우려사항": st.sidebar.slider("우려사항 민감도 (감점)", 1, 5, 3),
    "미래 잠재역량": st.sidebar.slider("미래 잠재력 중요도", 1, 5, 2),
}

# --- 업로드 및 옵션 ---
uploaded_files = st.file_uploader("📄 자기소개서 업로드 (PDF 또는 TXT)", type=["pdf", "txt"], accept_multiple_files=True)
email_enabled = st.checkbox("📧 리포트를 이메일로 발송하기")
email_address = st.text_input("수신 이메일 주소", value="") if email_enabled else None

# --- 텍스트 추출 함수 ---
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])
    return file.read().decode("utf-8")

# --- HTML 리포트 템플릿 ---
def generate_html_report(results):
    html = """
    <html><head><style>
    body { font-family: 'Nanum Gothic', sans-serif; margin: 40px; }
    h1 { color: #2F4F4F; }
    h2 { color: #4B9CD3; border-bottom: 1px solid #ccc; }
    .section { margin-bottom: 30px; }
    .label { font-weight: bold; color: #444; }
    ul { padding-left: 20px; }
    </style></head><body>
    <h1>지원자 분석 리포트</h1>
    """
    for res in results:
        html += f"<div class='section'><h2>{res['파일명']}</h2>"
        html += f"<p><span class='label'>적합도 점수:</span> {res['전반적 적합도 점수']}점</p>"
        html += f"<p><span class='label'>추천 여부:</span> {res['추천 여부']}</p>"
        html += f"<p><span class='label'>미래 잠재역량:</span> {res['미래 잠재역량 또는 성장 가능성']}</p>"
        html += "<p><span class='label'>핵심 경험 및 키워드:</span></p><ul>"
        html += "".join(f"<li>{item}</li>" for item in res["핵심 경험과 키워드"]) + "</ul>"
        html += "<p><span class='label'>강점:</span></p><ul>"
        html += "".join(f"<li>{item}</li>" for item in res["강점"]) + "</ul>"
        html += "<p><span class='label'>우려사항:</span></p><ul>"
        html += "".join(f"<li>{item}</li>" for item in res["우려사항"]) + "</ul>"
        html += "<p><span class='label'>역량별 평가 코멘트:</span></p><ul>"
        for k, v in res.get("역량별 평가 코멘트", {}).items():
            html += f"<li><b>{k}</b>: {v}</li>"
        html += "</ul>"
        html += f"<p><span class='label'>종합 의견 요약:</span> {res['종합 의견 요약']}</p><hr></div>"
    html += "</body></html>"
    return html

# --- 분석 실행 ---
results = []
if st.button("📊 적합도 분석 실행") and uploaded_files and jd_input:
    for file in uploaded_files:
        resume_text = extract_text(file)
        prompt = f"""
        JD 또는 기대사항: {jd_input}

        자기소개서:
        {resume_text}

        다음 항목에 대해 JSON 형식으로 상세히 분석해줘:
        {{
        "핵심 경험과 키워드": [...],
        "전반적 적합도 점수": 0~100 정수,
        "강점": [...],
        "우려사항": [...],
        "종합 의견 요약": "...",
        "추천 여부": "강력 추천 / 가능 / 보통 / 비추천",
        "미래 잠재역량 또는 성장 가능성": "...",
        "역량별 평가 코멘트": {{
            "문제 해결력": "...",
            "데이터 활용력": "...",
            "협업/커뮤니케이션": "...",
            "학습 및 성장의지": "..."
        }}
        }}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()
            json_data = content[content.find("{"):content.rfind("}") + 1]
            parsed = json.loads(json_data)
            parsed["파일명"] = file.name
            results.append(parsed)
        except Exception as e:
            st.error(f"❌ {file.name} 분석 오류: {e}")

# --- 결과 시각화 + 리포트 ---
if results:
    st.success("✅ 분석 완료")
    score_data = []
    for r in results:
        score = (
            r["전반적 적합도 점수"] * weights["핵심 경험과 키워드"] +
            len(r["강점"]) * weights["강점"] +
            weights["미래 잠재역량"] * 2 -
            len(r["우려사항"]) * weights["우려사항"]
        )
        score_data.append((r["파일명"], score))

    df = pd.DataFrame(score_data, columns=["지원자", "가중 점수"])
    st.plotly_chart(px.bar(df, x="지원자", y="가중 점수", color="지원자", text_auto=True), use_container_width=True)

    # PDF 리포트 생성
    html = generate_html_report(results)
    pdf_bytes = HTML(string=html).write_pdf()

    # 다운로드 버튼
    st.download_button("📥 디자인된 PDF 리포트 다운로드", data=pdf_bytes,
                       file_name="채용_분석_리포트.pdf", mime="application/pdf")

    # 이메일 전송
    if email_enabled and email_address:
        try:
            msg = EmailMessage()
            msg['Subject'] = '지원자 분석 리포트'
            msg['From'] = 'noreply@example.com'
            msg['To'] = email_address
            msg.set_content("채용 분석 리포트를 첨부드립니다.")
            msg.add_attachment(pdf_bytes, maintype='application', subtype='pdf', filename="채용_분석_리포트.pdf")
            with smtplib.SMTP('smtp.example.com', 587) as server:
                server.starttls()
                server.login('noreply@example.com', 'password')  # 실 계정 필요
                server.send_message(msg)
            st.success("📧 이메일 발송 완료!")
        except Exception as e:
            st.error(f"이메일 발송 실패: {e}")
