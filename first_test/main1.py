import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# CONFIG
# -------------------------------
AREA_NAME = "Al Khairan First"
MACRO_NEWS_FACTOR = 1.045  # Yas Island structural uplift

# -------------------------------
# LOAD ASSETS
# -------------------------------
@st.cache_resource
def load_model():
    with open("rf_model_Al Khairan First.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_columns():
    with open("trained_columns_Al Khairan First.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
training_columns = load_columns()

growth_df = pd.read_csv("Sarima_forecast_6M.csv")
growth_df = growth_df[growth_df["area_name_en"] == AREA_NAME]
growth_df["month"] = pd.to_datetime(growth_df["month"])

historical_df = pd.read_csv("historical_df.csv")
historical_df = historical_df[historical_df["area_name_en"] == AREA_NAME]
historical_df["month"] = pd.to_datetime(historical_df["month"])

# -------------------------------
# CATEGORIES (from your charts)
# -------------------------------
DEVELOPER_CATEGORIES = ["Grade 1", "Others"]

PROJECT_CATEGORIES = [
    "Developer-Led Mid-Market",
    "General / Unclassified",
    "Parks & Eco-Living",
    "Standard Residential",
    "Investment & High-Volume",
    "Urban High Rise",
    "Ultra-Luxury & Premium",
    "City Centric & Boulevards",
    "Waterfront & Marine",
    "Golf & Sports Lifestyle",
    "Suburban & Gated Communities",
    "Managed & Serviced",
    "Modern & Concept Living",
    "Scenic Views & Vistas"
]

FLOOR_BINS = ["0-10", "11-20", "21-30", "31-40", "41-50"]

ROOM_MAP = {
    "Studio": "Studio",
    "1 B/R": "1 B/R",
    "2 B/R": "2 B/R",
    "3 B/R": "3 B/R",
    "4 B/R": "4 B/R",
    "5 B/R": "5 B/R",
    "6 B/R": "6 B/R"
}

REG_TYPE_MAP = ["Off-Plan Properties", "Existing Properties "]

# -------------------------------
# INPUT PREPARATION
# -------------------------------
def prepare_input(user_vals):
    row = pd.DataFrame([user_vals])

    for col in training_columns:
        if col not in row.columns:
            row[col] = 0

    return row[training_columns]

def anchor_prediction(last_hist, pred, max_dev=0.06):
    """
    Force prediction to stay close to last historical value
    """
    upper = last_hist * (1 + max_dev)
    lower = last_hist * (1 - max_dev)
    return float(np.clip(pred, lower, upper))


def smooth_ewma(series, span=4):
    """
    Heavy smoothing using exponential moving average
    """
    return pd.Series(series).ewm(span=span, adjust=False).mean().values





# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("📈 Yas Island Meter Sale Price Forecast")

st.caption(
    "Forecast combines historical pricing, model-based valuation, "
    "organic market growth, and structural macro-event adjustments specific to Yas Island."
)

# -------------------------------
# USER INPUTS (LIMITED)
# -------------------------------
st.sidebar.header("Property Configuration")

rooms = st.sidebar.selectbox("Rooms", list(ROOM_MAP.keys()), index=2)
floor = st.sidebar.selectbox("Floor Range", FLOOR_BINS, index=1)

swimming_pool = st.sidebar.checkbox("Swimming Pool", value=True)
balcony = st.sidebar.checkbox("Balcony", value=True)
metro = st.sidebar.checkbox("Near Metro", value=False)

project_cat = st.sidebar.selectbox("Project Category", PROJECT_CATEGORIES)
developer_cat = st.sidebar.selectbox("Developer Category", DEVELOPER_CATEGORIES)

reg_type = st.sidebar.selectbox("Registration Type", REG_TYPE_MAP, index=1)

# -------------------------------
# BUILD MODEL INPUT (FROZEN + USER)
# -------------------------------
model_input = {
    "reg_type_en": reg_type,
    "rooms_en": ROOM_MAP.get(rooms, "2 B/R"),
    "has_parking": 1,
    "procedure_area": 80,
    "land_type_en": "Commercial",
    "floor_bin": floor if floor in FLOOR_BINS else "11-20",
    "swimming_pool": int(swimming_pool),
    "balcony": int(balcony),
    "elevator": 1,
    "metro": int(metro),
    "project_cat": project_cat,
    "developer_cat": developer_cat
}

X = prepare_input(model_input)
base_price = model.predict(X)[0]

# -------------------------------
# FORECAST CONSTRUCTION
# -------------------------------
forecast_df = growth_df.copy()
forecast_df["baseline_price"] = base_price * forecast_df["growth_factor"]
forecast_df["scenario_price (with macroeconomic features and news)"] = forecast_df["baseline_price"] * MACRO_NEWS_FACTOR


st.subheader("📊 Historical, Predicted & Forecasted Prices")

# -------------------------------
# 1. Smooth historical prices
# -------------------------------
hist_df = historical_df.sort_values("month")
hist_df["smooth_price"] = smooth_ewma(hist_df["median_price"], span=5)

last_hist_price = hist_df["smooth_price"].iloc[-1]
last_hist_date = hist_df["month"].iloc[-1]

# -------------------------------
# 2. Anchor prediction
# -------------------------------
anchored_prediction = anchor_prediction(
    last_hist=last_hist_price,
    pred=base_price,
    max_dev=0.06  # max 6% jump allowed
)

# -------------------------------
# 3. Rebase forecast from anchored prediction
# -------------------------------
forecast_df = forecast_df.sort_values("month")

baseline_raw = anchored_prediction * (
    forecast_df["baseline_price"] / forecast_df["baseline_price"].iloc[0]
)

scenario_raw = anchored_prediction * (
    forecast_df["scenario_price (with macroeconomic features and news)"] / forecast_df["baseline_price"].iloc[0]
)

forecast_df["baseline_smooth"] = smooth_ewma(baseline_raw, span=4)
forecast_df["scenario_price (with macroeconomic features and news)"] = smooth_ewma(scenario_raw, span=4)

# -------------------------------
# 4. Build Plotly figure
# -------------------------------
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=hist_df["month"],
    y=hist_df["smooth_price"],
    mode="lines",
    name="Historical Trend",
    line=dict(width=3)
))

# Prediction point (connected)
fig.add_trace(go.Scatter(
    x=[last_hist_date, forecast_df["month"].iloc[0]],
    y=[last_hist_price, anchored_prediction],
    mode="lines+markers",
    name="Model Prediction",
    line=dict(width=2, dash="dot"),
    marker=dict(size=10)
))

# Baseline forecast
fig.add_trace(go.Scatter(
    x=forecast_df["month"],
    y=forecast_df["baseline_smooth"],
    mode="lines",
    name="Baseline Forecast",
    line=dict(width=3, dash="dash")
))

# Scenario forecast
fig.add_trace(go.Scatter(
    x=forecast_df["month"],
    y=forecast_df["scenario_price (with macroeconomic features and news)"],
    mode="lines",
    name="Scenario Forecast",
    line=dict(width=4)
))

# -------------------------------
# 5. Styling
# -------------------------------
fig.update_layout(
    height=520,
    xaxis_title="Month",
    yaxis_title="Meter Sale Price (AED)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.update_xaxes(
    dtick="M1",
    tickformat="%b %Y",
    tickangle=-45,
    showgrid=True
)


# Shade forecast region
fig.add_vrect(
    x0=forecast_df["month"].min(),
    x1=forecast_df["month"].max(),
    fillcolor="lightblue",
    opacity=0.08,
    layer="below",
    line_width=0
)

st.plotly_chart(fig, use_container_width=True)



# -------------------------------
# EXPLANATION
# -------------------------------
st.subheader("🧠 Scenario Explanation")

st.write(f"**Base Model Price:** {base_price:,.0f} AED / m²")
st.write(f"**Structural Adjustment Applied:** +{(MACRO_NEWS_FACTOR-1)*100:.2f}%")

st.markdown(
    """
**Scenario assumptions include:**
- Major entertainment-led development in Yas Island (Disneyland theme destination)
- Tourism-driven demand expansion
- Supporting infrastructure and hospitality investments
- Stable regulatory and interest rate environment

These factors are applied as a structural uplift over the organic growth forecast.
"""
)

# -------------------------------
# DATA TABLE
# -------------------------------
with st.expander("📄 Forecast Data"):
    st.dataframe(
        forecast_df[["month", "baseline_price", "scenario_price (with macroeconomic features and news)"]]
        .rename(columns={
            "baseline_price": "Baseline Forecast",
            "scenario_price (with macroeconomic features and news)": "Scenario Forecast with macroeconomic and news data"
        })
    )
