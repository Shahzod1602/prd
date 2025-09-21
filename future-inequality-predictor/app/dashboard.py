import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from predictor import forecast

# OpenAI client (optional)
client = None
try:
    from openai import OpenAI
    if "OPENAI_API_KEY" in st.secrets:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None

st.set_page_config(page_title="Future Inequality Predictor", layout="wide")
st.title("🌍 Future Inequality Predictor")

# -------------------------------
# Indicators map: file, column, default_target
indicators = {
    "Economy": ("economy.csv", "gdp_growth", 100.0),
    "Education": ("education.csv", "literacy_rate", 100.0),
    "Health": ("health.csv", "life_expectancy", 80.0),
    "Climate": ("climate.csv", "avg_ecology", 100.0),
    "Infrastructure": ("infrastructure.csv", "internet_access", 95.0),
    "Governance": ("governance.csv", "clean_governance_index", 100),  # lower = better
    "Employment": ("employment.csv", "employment_rate", 90.0),
    "Gender Equality": ("gender.csv", "women_employment", 60.0)
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def find_col(df, name):
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    return None

def read_current_and_target(file, column):
    path = os.path.join(DATA_DIR, file)
    if not os.path.exists(path):
        return (None, None)
    df = pd.read_csv(path)
    year_col = find_col(df, 'year') or find_col(df, 'Year') or find_col(df, 'ds')
    metric_col = find_col(df, column)
    if year_col and metric_col:
        s = df[metric_col].dropna()
        cur = float(s.iloc[-1]) if not s.empty else None
    else:
        cur_col = find_col(df, 'current_value') or find_col(df, 'current') or find_col(df, 'value')
        cur = float(df[cur_col].iloc[-1]) if cur_col and not df[cur_col].isnull().all() else None

    target_col = find_col(df, 'target_value') or find_col(df, 'target')
    target = float(df[target_col].iloc[-1]) if target_col and not df[target_col].isnull().all() else None
    return (cur, target)

# -------------------------------
# 1️⃣ Uzbekistan Forecast Section
# -------------------------------
st.header("🇺🇿 Uzbekistan SDG Forecast")

years = st.slider("Forecast horizon (years)", 5, 15, 10, key="forecast_horizon")

if st.button("Generate Uzbekistan Forecast"):
    labels = []
    current_vals = []
    forecast_vals = []
    target_vals = []
    problems = []

    for name, (file, column, default_target) in indicators.items():
        cur, file_target = read_current_and_target(file, column)
        try:
            fdf = forecast(file, column, years)
            last_forecast = fdf.tail(1).iloc[0]['yhat']
            forecast_value = float(last_forecast)
            if cur is None:
                path = os.path.join(DATA_DIR, file)
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    metric_col = find_col(df, column)
                    if metric_col:
                        s = df[metric_col].dropna()
                        if not s.empty:
                            cur = float(s.iloc[-1])
            if cur is None:
                cur = forecast_value
        except FileNotFoundError:
            problems.append(f"{name}: file not found ({file}). Skipped.")
            continue
        except Exception as e:
            if cur is not None:
                forecast_value = float(cur)
            else:
                problems.append(f"{name}: {str(e)}")
                continue

        target = file_target if file_target is not None else default_target
        labels.append(name)
        current_vals.append(float(cur))
        forecast_vals.append(float(forecast_value))
        target_vals.append(float(target))

    if not labels:
        st.error("❌ Hech qanday indikator ishlay olmadi. data/ papkani tekshiring.")
    else:
        # Radar chart (Uzbekistan)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=current_vals, theta=labels, fill='toself', name='Current'))
        fig.add_trace(go.Scatterpolar(r=forecast_vals, theta=labels, fill='toself', name=f'Forecast +{years}y'))
        fig.add_trace(go.Scatterpolar(r=target_vals, theta=labels, fill='toself', name='Target'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True,
                          title=f"Uzbekistan: Current vs Forecast (after {years} years) vs Target")
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=labels, y=current_vals, name="Current"))
        fig_bar.add_trace(go.Bar(x=labels, y=forecast_vals, name=f"Forecast +{years}y"))
        fig_bar.add_trace(go.Bar(x=labels, y=target_vals, name="Target"))
        fig_bar.update_layout(barmode='group', title="Uzbekistan Indicators Comparison")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Line chart
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=labels, y=current_vals, mode='lines+markers', name="Current"))
        fig_line.add_trace(go.Scatter(x=labels, y=forecast_vals, mode='lines+markers', name=f"Forecast +{years}y"))
        fig_line.add_trace(go.Scatter(x=labels, y=target_vals, mode='lines+markers', name="Target"))
        fig_line.update_layout(title="Uzbekistan Indicators Trend (Category-wise)")
        st.plotly_chart(fig_line, use_container_width=True)

        if problems:
            st.warning("⚠️ Some indicators had issues:")
            for p in problems:
                st.write("-", p)

        # AI recommendations
        if client is not None:
            try:
                prompt = f"""
You are a world-class geopolitical analyst and futurist. Analyze the future of a country based on its current economy, politics, culture, demographics, technological development, natural resources, and global position. Provide a highly detailed, personalized prediction covering: economic growth, political stability, technological advancements, international influence, social progress, and potential challenges. Include surprising insights and strategic recommendations that the country could adopt to thrive. Present your answer in a storytelling style that makes the reader amazed and deeply engaged. Uzbekistan indicators (Current, Forecast, Target) for {years} years:
{[{ 'indicator': labels[i], 'current': current_vals[i], 'forecast': forecast_vals[i], 'target': target_vals[i]} for i in range(len(labels))]}
Give: 1) top 3 areas at risk, 2) 3 priority policies, 3) quick 2-line summary for hackathon slide.
"""
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":"You are an SDG and policy expert."},
                              {"role":"user","content":prompt}]
                )
                advice = response.choices[0].message.content
                st.subheader("🤖 AI Strategic Recommendations (Uzbekistan)")
                st.write(advice)
            except Exception as e:
                st.error(f"AI call error: {e}")
        else:
            st.info("ℹ️ OpenAI key topilmadi → Uzbekistan AI recommendations o‘chirib qo‘yildi.")

# -------------------------------
# 2️⃣ Personal Analysis Section with 10-point Personal Factors
# -------------------------------
st.header("👤 Personal Progress Analysis")

st.sidebar.header("Enter Your Personal Information")
age = st.sidebar.number_input("Age", 10, 100, 20)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
education = st.sidebar.selectbox("Education Level", ["Primary", "Secondary", "Bachelor", "Master", "PhD"])
family_status = st.sidebar.selectbox("Family Status", ["Single", "Married", "With Children"])
income = st.sidebar.slider("Monthly Income (USD)", 0, 5000, 500)

st.sidebar.header("Rate Your Personal Factors (1-10)")
personal_factors = {
    "Financial Intelligence": st.sidebar.slider("Financial Intelligence", 1, 10, 5),
    "Hard Work & Discipline": st.sidebar.slider("Hard Work & Discipline", 1, 10, 5),
    "Analytical & Strategic Thinking": st.sidebar.slider("Analytical & Strategic Thinking", 1, 10, 5),
    "Risk Management": st.sidebar.slider("Risk Management", 1, 10, 5),
    "Innovative Thinking": st.sidebar.slider("Innovative Thinking", 1, 10, 5),
    "Networking / Connections": st.sidebar.slider("Networking / Connections", 1, 10, 5),
    "Community & Environment": st.sidebar.slider("Community & Environment", 1, 10, 5),
    "Market Trends & Opportunities": st.sidebar.slider("Market Trends & Opportunities", 1, 10, 5),
    "Country Economy & Legislation": st.sidebar.slider("Country Economy & Legislation", 1, 10, 5),
    "Serendipitous Opportunities": st.sidebar.slider("Serendipitous Opportunities", 1, 10, 5)
}

if st.button("Analyze Personal Progress"):
    # Normalize 1-10 to 0-100
    personal_scores = [v * 10 for v in personal_factors.values()]
    labels = list(personal_factors.keys())

    # Radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=personal_scores, theta=labels, fill='toself', name="You"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      showlegend=True, title="Personal Factors Radar")
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=labels, y=personal_scores, name="Your Scores"))
    fig_bar.update_layout(title="Personal Factors (Bar Chart)", yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Line chart
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=labels, y=personal_scores, mode='lines+markers', name="You"))
    fig_line.update_layout(title="Personal Factors Trend", yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_line, use_container_width=True)

    # AI Recommendations
    if client is not None:
        try:
            prompt = f"""
You are a life coach AI.
User info: Age={age}, Gender={gender}, Education={education}, Family={family_status}, Income={income}.
Personal factors (1-10 scale): {personal_factors}

1) Identify 3 strengths and 3 weaknesses for this person.
2) Suggest 3 actionable steps to improve personal growth.
3) Give a 2-line motivational summary.
"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"You are a world-class life analyst and futurist. Analyze the future of a person based on their current traits, habits, skills, interests, and environment. Provide a highly detailed, personalized prediction that covers: career potential, financial success, personal relationships, intellectual growth, and overall life path. Include unexpected insights that will surprise the user, and suggest specific actions they can take to maximize their future potential. Present your answer in a storytelling style that makes the user feel amazed and inspired"},
                          {"role":"user","content":prompt}]
            )
            advice = response.choices[0].message.content
            st.subheader("🤖 AI Personalized Recommendations")
            st.write(advice)
        except Exception as e:
            st.error(f"AI call error: {e}")
    else:
        st.info("ℹ️ OpenAI key topilmadi → Personal AI recommendations o‘chirib qo‘yildi.")
