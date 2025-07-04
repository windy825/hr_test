import streamlit as st
import openai
import PyPDF2
import pandas as pd
import plotly.express as px
import json

st.set_page_config(page_title="채용 적합도 분석기", layout="wide")
st.title("✨ GPT 기반 채용 적합도 분석기")

# API 입력
st.sidebar.title("🔐 GPT API Key")
api_key = st.sidebar.text_input("OpenAI API Key 입력", type="password")
if not api_key:
    st.warning("🔑 API Key를 입력해주세요.")
    st.stop()
client = openai.OpenAI(api_key=api_key)

# JD + 가중치 설정
st.sidebar.subheader("📌 JD 입력")
jd_input = st.sidebar.text_area("JD 또는 인사담당자 메모")

st.sidebar.subheader("⚖️ JD 중요도 가중치")
weights = {
    "핵심 경험과 키워드": st.sidebar.slider("경험 키워드 중요도", 1, 5, 3),
    "강점": st.sidebar.slider("강점 항목 중요도", 1, 5, 3),
    "우려사항": st.sidebar.slider("우려사항 민감도 (감점)", 1, 5, 3),
    "미래 잠재역량": st.sidebar.slider("미래 잠재력 중요도", 1, 5, 2),
}

# 파일 업로드
uploaded_files = st.file_uploader("📄 자기소개서 업로드 (PDF 또는 TXT)", type=["pdf", "txt"], accept_multiple_files=True)

def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])
    return file.read().decode("utf-8")

# HTML 리포트 템플릿
def generate_html_report(results):
    html = """
    <html><head><style>
    body { font-family: 'Nanum Gothic', sans-serif; margin: 40px; line-height: 1.6; }
    h1 { color: #2F4F4F; }
    h2 { color: #4B9CD3; border-bottom: 1px solid #ccc; padding-bottom: 5px; }
    .section { margin-bottom: 40px; }
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

if results:
    st.success("✅ 분석 완료")

    # 점수 계산 및 시각화
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

    # HTML 리포트 렌더링
    html = generate_html_report(results)
    st.markdown("## 🖨️ 리포트 보기 및 PDF 저장 안내")
    st.components.v1.html(html, height=1000, scrolling=True)
    st.info("📄 리포트를 PDF로 저장하려면 브라우저에서 Ctrl+P (또는 ⌘+P) 를 눌러 'PDF로 저장'을 선택하세요.")
