

import streamlit as st
import pandas as pd
import plotly.express as px

# General information (example, you can replace with your CSV values)
data = {
    "Indicator": ["Economy", "Education", "Health", "Climate"],
    "Current Value": [4.2, 97, 72, 16.5],   # misol uchun: GDP %, Literacy %, Life expectancy, Avg temp
    "Target Value": [6.0, 100, 80, 15.0]   # ideal yoki hukumat target
}

df = pd.DataFrame(data)

# Radar chart (current vs target)
fig = px.line_polar(df, r="Current Value", theta="Indicator", line_close=True,
                    title="Uzbekistan General Indicators (Radar View)")
fig.add_scatterpolar(r=df["Target Value"], theta=df["Indicator"], line_close=True, name="Target")

st.plotly_chart(fig)
