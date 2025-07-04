import streamlit as st
import openai
import PyPDF2
import pandas as pd
import plotly.express as px
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go

# 페이지 설정
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

# 이력서 업로드
uploaded_files = st.file_uploader("📄 자기소개서 업로드 (PDF 또는 TXT)", type=["pdf", "txt"], accept_multiple_files=True)

# 텍스트 추출 함수
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])
    return file.read().decode("utf-8")

# HTML 보고서 생성 함수
def generate_detailed_html_report(results):
    html = """
    <html><head><style>
    body { font-family: 'Nanum Gothic', sans-serif; margin: 30px; line-height: 1.7; }
    h1 { color: #1F4E79; }
    h2 { color: #336699; border-bottom: 2px solid #ddd; padding-bottom: 6px; margin-top: 40px; }
    .section { margin-bottom: 60px; }
    .label { font-weight: bold; color: #333; }
    .score-box { font-size: 1.2em; color: #2E8B57; margin: 10px 0; }
    ul { padding-left: 20px; }
    .radar-chart, .wordcloud { margin-top: 15px; margin-bottom: 30px; }
    </style></head><body>
    <h1>지원자 채용 분석 리포트</h1>
    """
    for res in results:
        soft_skills = res.get("역량별 평가 코멘트", {})

        # WordCloud 생성
        keywords_text = " ".join(res.get("핵심 경험과 키워드", []))
        wordcloud = WordCloud(font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                              background_color="white", width=600, height=300).generate(keywords_text)
        wc_buffer = BytesIO()
        plt.figure(figsize=(6,3))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(wc_buffer, format="png")
        plt.close()
        wc_data = base64.b64encode(wc_buffer.getvalue()).decode("utf-8")

        # Radar Chart 생성
        radar_labels = list(soft_skills.keys())
        radar_values = [5 if "우수" in v or "높음" in v else 3 if "보통" in v else 1 for v in soft_skills.values()]
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(r=radar_values, theta=radar_labels, fill='toself'))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False)
        radar_buffer = BytesIO()
        radar_fig.write_image(radar_buffer, format="png", width=500, height=400)
        radar_data = base64.b64encode(radar_buffer.getvalue()).decode("utf-8")

        html += f"<div class='section'><h2>{res['파일명']}</h2>"
        html += f"<p class='score-box'>✅ 적합도 점수: <strong>{res['전반적 적합도 점수']}</strong>점 | 추천 여부: <strong>{res['추천 여부']}</strong></p>"
        html += f"<p><span class='label'>미래 잠재역량:</span> {res['미래 잠재역량 또는 성장 가능성']}</p>"

        html += "<h3>📌 핵심 경험 및 키워드</h3><ul>" + "".join(f"<li>{kw}</li>" for kw in res["핵심 경험과 키워드"]) + "</ul>"
        html += f"<div class='wordcloud'><img src='data:image/png;base64,{wc_data}' width='100%'></div>"

        html += "<h3>💪 강점</h3><ul>" + "".join(f"<li>{g}</li>" for g in res["강점"]) + "</ul>"
        html += "<h3>⚠️ 우려사항</h3><ul>" + "".join(f"<li>{w}</li>" for w in res["우려사항"]) + "</ul>"

        html += "<h3>🧠 역량별 평가</h3><ul>"
        for k, v in soft_skills.items():
            html += f"<li><b>{k}</b>: {v}</li>"
        html += "</ul>"

        html += f"<div class='radar-chart'><img src='data:image/png;base64,{radar_data}'></div>"
        html += f"<h3>📝 종합 의견 요약</h3><p>{res['종합 의견 요약']}</p></div>"

    html += "</body></html>"
    return html

# GPT 분석 시작
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

# 결과 시각화 및 HTML 보고서 출력
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

    html = generate_detailed_html_report(results)
    st.markdown("## 🖨️ 리포트 보기 및 PDF 저장 안내")
    st.components.v1.html(html, height=1800, scrolling=True)
    st.info("📄 Ctrl+P 또는 ⌘+P 를 눌러 PDF로 저장하세요.")
