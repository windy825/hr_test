import streamlit as st
import openai
import PyPDF2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import json
import base64
import numpy as np
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
import os, requests

# 📌 기본 설정
st.set_page_config(page_title="채용 적합도 분석기", layout="wide")
st.title("✨ GPT 기반 채용 적합도 분석기")

# 🔐 API 입력
st.sidebar.title("🔐 GPT API Key")
api_key = st.sidebar.text_input("OpenAI API Key 입력", type="password")
if not api_key:
    st.warning("🔑 API Key를 입력해주세요.")
    st.stop()
client = openai.OpenAI(api_key=api_key)

# 📌 JD + 가중치
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

def to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    return base64.b64encode(buf.getvalue()).decode()

# ✅ WordCloud (폰트 자동 다운로드)
def generate_wordcloud(text):
    font_path = "/tmp/NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/naver/nanumfont/releases/download/VER2.5/NanumGothic.ttf"
        r = requests.get(url)
        with open(font_path, "wb") as f:
            f.write(r.content)
    wc = WordCloud(font_path=font_path, background_color="white", width=600, height=300).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return to_base64(fig)

# ✅ Radar Chart
def generate_radar_chart(labels, values):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False)
    buf = BytesIO()
    fig.write_image(buf, format="png", width=500, height=400)
    return base64.b64encode(buf.getvalue()).decode()

# ✅ 종합 요약 시각화
def generate_summary_charts(results):
    names = [r["파일명"] for r in results]
    scores = [r["전반적 적합도 점수"] for r in results]
    strengths = [len(r["강점"]) for r in results]
    weaknesses = [len(r["우려사항"]) for r in results]

    # 히트맵
    df = pd.DataFrame({
        "파일명": names,
        "적합도": scores,
        "강점": strengths,
        "우려사항": weaknesses
    }).set_index("파일명")
    normed = MinMaxScaler().fit_transform(df)
    fig1, ax1 = plt.subplots()
    sns.heatmap(normed, annot=df.values, fmt=".0f", cmap="YlGnBu", xticklabels=df.columns, yticklabels=names, ax=ax1)
    heatmap_b64 = to_base64(fig1)

    # 평균 역량 점수
    all_scores = {}
    for r in results:
        for k, v in r["역량별 평가 코멘트"].items():
            score = 5 if "우수" in v or "높음" in v else 3 if "보통" in v else 1
            all_scores[k] = all_scores.get(k, 0) + score
    avg_scores = {k: round(v / len(results), 2) for k, v in all_scores.items()}
    fig2, ax2 = plt.subplots()
    sns.barplot(x=list(avg_scores.keys()), y=list(avg_scores.values()), ax=ax2)
    avg_b64 = to_base64(fig2)

    return heatmap_b64, avg_b64

# ✅ GPT 분석 실행
results = []
if st.button("📊 적합도 분석 실행") and uploaded_files and jd_input:
    for file in uploaded_files:
        text = extract_text(file)
        prompt = f"""
        JD 또는 기대사항: {jd_input}

        자기소개서:
        {text}

        다음 항목에 대해 JSON 형식으로 분석해줘:
        {{
        "핵심 경험과 키워드": [...],
        "전반적 적합도 점수": 0~100 정수,
        "강점": [...],
        "우려사항": [...],
        "종합 의견 요약": "...",
        "추천 여부": "...",
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
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            content = res.choices[0].message.content
            parsed = json.loads(content[content.find("{"):content.rfind("}") + 1])
            parsed["파일명"] = file.name
            results.append(parsed)
        except Exception as e:
            st.error(f"{file.name} 분석 실패: {e}")

# ✅ HTML 보고서 렌더링
if results:
    st.success("✅ 분석 완료")
    st.markdown("## 📑 분석 리포트 (PDF 저장 가능)")

    html = "<html><body><h1>채용 적합도 분석 리포트</h1>"
    for r in results:
        radar_labels = list(r["역량별 평가 코멘트"].keys())
        radar_values = [5 if "우수" in v or "높음" in v else 3 if "보통" in v else 1 for v in r["역량별 평가 코멘트"].values()]
        wordcloud_b64 = generate_wordcloud(" ".join(r["핵심 경험과 키워드"]))
        radar_b64 = generate_radar_chart(radar_labels, radar_values)

        html += f"<h2>{r['파일명']}</h2>"
        html += f"<p><b>적합도 점수:</b> {r['전반적 적합도 점수']} | <b>추천:</b> {r['추천 여부']}</p>"
        html += f"<p><b>미래 잠재역량:</b> {r['미래 잠재역량 또는 성장 가능성']}</p>"
        html += "<h4>📌 핵심 경험 및 키워드</h4><ul>" + "".join(f"<li>{kw}</li>" for kw in r["핵심 경험과 키워드"]) + "</ul>"
        html += f"<img src='data:image/png;base64,{wordcloud_b64}' width='600'/>"
        html += "<h4>💪 강점</h4><ul>" + "".join(f"<li>{s}</li>" for s in r["강점"]) + "</ul>"
        html += "<h4>⚠️ 우려사항</h4><ul>" + "".join(f"<li>{w}</li>" for w in r["우려사항"]) + "</ul>"
        html += "<h4>🧠 역량별 평가</h4><ul>" + "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in r["역량별 평가 코멘트"].items()) + "</ul>"
        html += f"<img src='data:image/png;base64,{radar_b64}' width='500'/>"
        html += f"<h4>📝 종합 의견</h4><p>{r['종합 의견 요약']}</p><hr>"

    heatmap_b64, avg_b64 = generate_summary_charts(results)
    html += "<h2>📊 전체 지원자 종합 분석</h2>"
    html += f"<h4>지원자별 지표 히트맵</h4><img src='data:image/png;base64,{heatmap_b64}' width='700'/>"
    html += f"<h4>역량 평균 점수</h4><img src='data:image/png;base64,{avg_b64}' width='600'/>"
    html += "</body></html>"

    st.components.v1.html(html, height=2200, scrolling=True)
    st.info("💾 PDF 저장: 브라우저에서 Ctrl+P 또는 ⌘+P를 눌러 'PDF로 저장' 선택")
