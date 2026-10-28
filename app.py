import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Causal Pricing Engine", layout="wide")

st.title("📈 Algorithmic Pricing Causal Engine")
st.markdown("### Executive Dashboard — Synthetic Control & Double Machine Learning")

@st.cache_data
def load_data():
    df = pd.read_csv('data/market_telemetry.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

st.sidebar.header("Filter Telemetry")
selected_region = st.sidebar.selectbox("Select Market Region", df['region'].unique())

filtered_df = df[df['region'] == selected_region]

fig = px.line(filtered_df, x='date', y='revenue', title=f"Historical Revenue Trajectory — {selected_region}")
st.plotly_chart(fig, use_container_width=True)