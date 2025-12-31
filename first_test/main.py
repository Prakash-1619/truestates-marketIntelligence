import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import os


# -------------------------------
# PAGE CONFIG (MUST BE FIRST)
# -------------------------------
st.set_page_config(
    page_title="Macro based Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)
tab = st.sidebar.radio( "Zone", ["Yas Island", "Dubai South"])

# -------------------------------

# Tab 1

# -------------------------------
#BASE_DIR = os.path.dirname(os.path.abspath('first_test/'))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if tab == "Yas Island":
    # -------------------------------
    # CONFIG
    # -------------------------------
    AREA_NAME = "Al Khairan First"
    MACRO_NEWS_FACTOR = 1.045
    
    
    # -------------------------------
    # LOAD ASSETS
    # -------------------------------
    @st.cache_resource
    def load_model():
        model_path = os.path.join(BASE_DIR, "rf_model_Al Khairan First.pkl")
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
    @st.cache_resource
    def load_columns():
        col_path = os.path.join(BASE_DIR, "trained_columns_Al Khairan First.pkl")
        with open(col_path, "rb") as f:
            return pickle.load(f)
    
    
    model = load_model()
    training_columns = load_columns()
    
    growth_df = pd.read_csv(
        os.path.join(BASE_DIR, "Sarima_forecast_6M.csv")
    )
    
    historical_df = pd.read_csv(
        os.path.join(BASE_DIR, "historical_df.csv")
    )
    
    #growth_df = pd.read_csv("Sarima_forecast_6M.csv")
    growth_df = growth_df[growth_df["area_name_en"] == AREA_NAME]
    growth_df["month"] = pd.to_datetime(growth_df["month"])
    
    #historical_df = pd.read_csv("historical_df.csv")
    historical_df = historical_df[historical_df["area_name_en"] == AREA_NAME]
    historical_df["month"] = pd.to_datetime(historical_df["month"])
    
    # -------------------------------
    # HISTORICAL MONTHLY AGGREGATION
    # -------------------------------
    monthly_hist_df = (
        historical_df
        .groupby(pd.Grouper(key="month", freq="M"))
        .agg(
            median_price=("median_price", "median")
        )
        .reset_index()
    )
    
    # Interpolate missing median_price to ensure line connectivity
    monthly_hist_df["median_price"] = monthly_hist_df["median_price"].interpolate(method='linear')
    
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
        upper = last_hist * (1 + max_dev)
        lower = last_hist * (1 - max_dev)
        return float(np.clip(pred, lower, upper))
    
    # -------------------------------
    # UI
    # -------------------------------
    st.title("📈 Yas Island Meter Sale Price Forecast")
    st.caption(
        "Forecast combines historical pricing, model-based valuation, "
        "organic market growth, and structural macro-event adjustments specific to Yas Island."
    )
    
    # -------------------------------
    # USER INPUTS
    # -------------------------------
    ROOM_MAP = {
        "Studio": "Studio",
        "1 B/R": "1 B/R",
        "2 B/R": "2 B/R",
        "3 B/R": "3 B/R",
        "4 B/R": "4 B/R",
        "5 B/R": "5 B/R",
        "6 B/R": "6 B/R"
    }
    
    FLOOR_BINS = ["0-10", "11-20", "21-30", "31-40", "41-50"]
    
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
    
    REG_TYPE_MAP = ["Off-Plan Properties", "Existing Properties "]
    
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
    # MODEL INPUT & BASE PRICE
    # -------------------------------
    model_input = {
        "reg_type_en": reg_type,
        "rooms_en": ROOM_MAP.get(rooms, "2 B/R"),
        "has_parking": 1,
        "procedure_area": 80,
        "land_type_en": "Commercial",
        "floor_bin": floor,
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
    forecast_df = growth_df.copy().sort_values("month")
    forecast_df["baseline_price"] = base_price * forecast_df["growth_factor"]
    forecast_df["scenario_price"] = forecast_df["baseline_price"] * MACRO_NEWS_FACTOR
    
    # -------------------------------
    # HISTORICAL DATA
    # -------------------------------
    hist_df = monthly_hist_df.copy()
    last_hist_price = hist_df["median_price"].dropna().iloc[-1]
    last_hist_date = hist_df["month"].max()
    
    # -------------------------------
    # ANCHOR PREDICTION
    # -------------------------------
    anchored_prediction = anchor_prediction(last_hist_price, base_price)
    
    # -------------------------------
    # REPLACE FIRST FORECAST VALUE WITH PREDICTION
    # -------------------------------
    forecast_df = forecast_df.reset_index(drop=True)
    forecast_df.loc[0, "baseline_price"] = anchored_prediction
    forecast_df.loc[0, "scenario_price"] = anchored_prediction
    
    # -------------------------------
    # PLOT
    # -------------------------------
    fig = go.Figure()
    
    # 1. Historical line (fully connected)
    fig.add_trace(go.Scatter(
        x=hist_df["month"],
        y=hist_df["median_price"],
        mode="lines+markers",
        name="Historical",
        line=dict(width=3, color="Blue"),
        marker=dict(size=6)
    ))
    
    # 2. Prediction point highlighted
    fig.add_trace(go.Scatter(
        x=[last_hist_date],
        y=[anchored_prediction],
        mode="markers+text",
        name="Prediction",
        marker=dict(color="red", size=12, symbol="diamond"),
        text=["Prediction"],
        textposition="top center"
    ))
    
    # 3. Baseline forecast connected from prediction
    baseline_line_x = [last_hist_date] + list(forecast_df["month"][1:])
    baseline_line_y = [anchored_prediction] + list(forecast_df["baseline_price"][1:])
    fig.add_trace(go.Scatter(
        x=baseline_line_x,
        y=baseline_line_y,
        mode="lines",
        name="Baseline Forecast",
        line=dict(width=3, dash="dash", color="yellow")
    ))
    
    # 4. Scenario forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["month"],
        y=forecast_df["scenario_price"],
        mode="lines",
        name="Scenario Forecast",
        line=dict(width=4, dash="dot", color="green")
    ))
    
    # -------------------------------
    # STYLING
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
        dtick="M12",
        tickformat="%Y",
        tickangle=-45,
        showgrid=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # -------------------------------
    # EXPLANATION
    # -------------------------------
    st.subheader("🧠 Scenario Explanation")
    st.write(f"**Base Model Price:** {base_price:,.0f} AED / m²")
    st.write(f"**Structural Adjustment Applied:** +{(MACRO_NEWS_FACTOR-1)*100:.2f}%")
    st.markdown("""
    **Scenario assumptions include:**
    - Major entertainment-led development in Yas Island
    - Tourism-driven demand expansion
    - Supporting infrastructure and hospitality investments
    - Stable regulatory and interest rate environment
    """)
    
    # -------------------------------
    # DATA TABLE
    # -------------------------------
    with st.expander("📄 Forecast Data"):
        st.dataframe(
            forecast_df[["month", "baseline_price", "scenario_price"]]
            .rename(columns={
                "baseline_price": "Baseline Forecast",
                "scenario_price": "Scenario Forecast"
            })
        )





if tab == "Dubai South":
    # -------------------------------
    # CONFIG
    # -------------------------------
    AREA_NAME = "Jabal Ali First"
    MACRO_NEWS_FACTOR = 1.077
    
    @st.cache_resource
    def load_model():
        model_path = os.path.join(BASE_DIR, "rf_model_Jabal Ali First.pkl")
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
    @st.cache_resource
    def load_columns():
        col_path = os.path.join(BASE_DIR, "trained_columns_Jabal Ali First.pkl")
        with open(col_path, "rb") as f:
            return pickle.load(f)
    
    
    model = load_model()
    training_columns = load_columns()
    
    growth_df = pd.read_csv(os.path.join(BASE_DIR, "Sarima_forecast_6M.csv"))
    historical_df = pd.read_csv(os.path.join(BASE_DIR, "historical_df.csv"))
    
    
    #growth_df = pd.read_csv("Sarima_forecast_6M.csv")
    growth_df = growth_df[growth_df["area_name_en"] == AREA_NAME]
    growth_df["month"] = pd.to_datetime(growth_df["month"])
    
    #historical_df = pd.read_csv("historical_df.csv")
    historical_df = historical_df[historical_df["area_name_en"] == AREA_NAME]
    historical_df["month"] = pd.to_datetime(historical_df["month"])
    
    # -------------------------------
    # HISTORICAL MONTHLY AGGREGATION
    # -------------------------------
    monthly_hist_df = (
        historical_df
        .groupby(pd.Grouper(key="month", freq="M"))
        .agg(
            median_price=("median_price", "median")
        )
        .reset_index()
    )
    
    # Interpolate missing median_price to ensure line connectivity
    monthly_hist_df["median_price"] = monthly_hist_df["median_price"].interpolate(method='linear')
    
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
        upper = last_hist * (1 + max_dev)
        lower = last_hist * (1 - max_dev)
        return float(np.clip(pred, lower, upper))
    
    # -------------------------------
    # UI
    # -------------------------------
    st.title("📈 Dubai South Meter Sale Price Forecast")
    st.caption(
        "Forecast combines historical pricing, model-based valuation, "
        "organic market growth, and structural macro-event adjustments specific to Yas Island."
    )
    
    # -------------------------------
    # USER INPUTS
    # -------------------------------
    ROOM_MAP = {
        "Studio": "Studio",
        "1 B/R": "1 B/R",
        "2 B/R": "2 B/R",
        "3 B/R": "3 B/R",
        "4 B/R": "4 B/R",
        "5 B/R": "5 B/R",
        "6 B/R": "6 B/R"
    }
    
    FLOOR_BINS = ["0-10", "11-20", "21-30", "31-40", "41-50"]
    
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
    
    REG_TYPE_MAP = ["Off-Plan Properties", "Existing Properties "]
    
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
    # MODEL INPUT & BASE PRICE
    # -------------------------------
    model_input = {
        "reg_type_en": reg_type,
        "rooms_en": ROOM_MAP.get(rooms, "2 B/R"),
        "has_parking": 1,
        "procedure_area": 80,
        "land_type_en": "Commercial",
        "floor_bin": floor,
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
    forecast_df = growth_df.copy().sort_values("month")
    forecast_df["baseline_price"] = base_price * forecast_df["growth_factor"]
    forecast_df["scenario_price"] = forecast_df["baseline_price"] * MACRO_NEWS_FACTOR
    
    # -------------------------------
    # HISTORICAL DATA
    # -------------------------------
    hist_df = monthly_hist_df.copy()
    last_hist_price = hist_df["median_price"].dropna().iloc[-1]
    last_hist_date = hist_df["month"].max()
    
    # -------------------------------
    # ANCHOR PREDICTION
    # -------------------------------
    anchored_prediction = anchor_prediction(last_hist_price, base_price)
    
    # -------------------------------
    # REPLACE FIRST FORECAST VALUE WITH PREDICTION
    # -------------------------------
    forecast_df = forecast_df.reset_index(drop=True)
    forecast_df.loc[0, "baseline_price"] = anchored_prediction
    forecast_df.loc[0, "scenario_price"] = anchored_prediction
    
    # -------------------------------
    # PLOT
    # -------------------------------
    fig = go.Figure()
    
    # 1. Historical line (fully connected)
    fig.add_trace(go.Scatter(
        x=hist_df["month"],
        y=hist_df["median_price"],
        mode="lines+markers",
        name="Historical",
        line=dict(width=3, color="Blue"),
        marker=dict(size=6)
    ))
    
    # 2. Prediction point highlighted
    fig.add_trace(go.Scatter(
        x=[last_hist_date],
        y=[anchored_prediction],
        mode="markers+text",
        name="Prediction",
        marker=dict(color="red", size=12, symbol="diamond"),
        text=["Prediction"],
        textposition="top center"
    ))
    
    # 3. Baseline forecast connected from prediction
    baseline_line_x = [last_hist_date] + list(forecast_df["month"][1:])
    baseline_line_y = [anchored_prediction] + list(forecast_df["baseline_price"][1:])
    fig.add_trace(go.Scatter(
        x=baseline_line_x,
        y=baseline_line_y,
        mode="lines",
        name="Baseline Forecast",
        line=dict(width=3, dash="dash", color="yellow")
    ))
    
    # 4. Scenario forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["month"],
        y=forecast_df["scenario_price"],
        mode="lines",
        name="Scenario Forecast",
        line=dict(width=4, dash="dot", color="green")
    ))
    
    # -------------------------------
    # STYLING
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
        dtick="M12",
        tickformat="%Y",
        tickangle=-45,
        showgrid=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # -------------------------------
    # EXPLANATION
    # -------------------------------
    st.subheader("🧠 Scenario Explanation")
    st.write(f"**Base Model Price:** {base_price:,.0f} AED / m²")
    st.write(f"**Structural Adjustment Applied:** +{(MACRO_NEWS_FACTOR-1)*100:.2f}%")
    st.markdown("""
    Scenario assumptions include:
    -Strategic growth driven by Al Maktoum International Airport expansion
    - Strong demand from aviation, logistics, and industrial-led employment
    - Residential absorption supported by affordable housing and long-term end-user demand
    - Ongoing infrastructure development and connectivity (roads, logistics corridors, metro extensions)
    - Stable regulatory framework supporting foreign investment and long-term visas
    - Gradual tourism and business spillover from Expo legacy developments
    """)
    

    
    
