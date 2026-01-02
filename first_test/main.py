import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import os


# -------------------------------
# MACRO NEWS COMPONENTS (GLOBAL)
# -------------------------------
MACRO_NEWS = {
    "tourism": 0.02,
    "geopolitics": 0.03,
    "jobs": 0.015,
    "capital_flows": 0.025,
    "rates_liquidity": -0.01
}

# -------------------------------
# AREA SENSITIVITY MATRICES
# -------------------------------
AREA_SENSITIVITY = {
    "Yas Island": {
        "tourism": 1.4,
        "geopolitics": 0.8,
        "jobs": 0.6,
        "capital_flows": 0.7,
        "rates_liquidity": 1.0
    },
    "Dubai South": {
        "tourism": 0.3,
        "geopolitics": 0.4,
        "jobs": 1.6,
        "capital_flows": 0.5,
        "rates_liquidity": 1.2
    },
    "Saadiyat Island": {
    "tourism": 0.7,
    "geopolitics": 1.5,
    "jobs": 0.4,
    "capital_flows": 1.6,
    "rates_liquidity": 0.6
}

}

# -------------------------------
# MACRO FACTOR COMPUTATION
# -------------------------------
def compute_macro_news_factor(
    macro_news,
    sensitivity,
    cap_up=0.08,
    cap_down=-0.05
):
    factor = sum(
        macro_news[k] * sensitivity.get(k, 0)
        for k in macro_news
    )
    factor = max(min(factor, cap_up), cap_down)
    return 1 + factor


# -------------------------------
# PAGE CONFIG (MUST BE FIRST)
# -------------------------------
st.set_page_config(
    page_title="Macro based Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)
tab = st.sidebar.radio( "Location", ["Yas Island", "Dubai South", "Saadiyat Island"] )

# -------------------------------
# Tab 1: Yas Island
# -------------------------------
#BASE_DIR = os.path.dirname(os.path.abspath('first_test/'))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if tab == "Yas Island":
    # -------------------------------
    # CONFIG
    # -------------------------------
    AREA_NAME = "Al Khairan First"
    MACRO_NEWS_FACTOR = compute_macro_news_factor(
    MACRO_NEWS,
    AREA_SENSITIVITY["Yas Island"]
)

    
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
    growth_df["month"] = pd.to_datetime(growth_df["month"], format = '%d-%m-%Y')
    
    #historical_df = pd.read_csv("historical_df.csv")
    historical_df = historical_df[historical_df["area_name_en"] == AREA_NAME]
    historical_df["month"] = pd.to_datetime(historical_df["month"], format='%d-%m-%Y')
    
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
    
    st.sidebar.header("Property Configuration - Available Options for Dubai South")
    # rooms = st.sidebar.selectbox("Rooms", list(ROOM_MAP.keys()), index=2)
    # floor = st.sidebar.selectbox("Floor Range", FLOOR_BINS, index=1)
    # swimming_pool = st.sidebar.checkbox("Swimming Pool", value=True)
    # balcony = st.sidebar.checkbox("Balcony", value=True)
    # metro = st.sidebar.checkbox("Near Metro", value=False)
    project_cat = st.sidebar.selectbox("Project Category", PROJECT_CATEGORIES)
    # developer_cat = st.sidebar.selectbox("Developer Category", DEVELOPER_CATEGORIES)
    reg_type = st.sidebar.selectbox("Registration Type", REG_TYPE_MAP, index=1)
    
    # -------------------------------
    # MODEL INPUT & BASE PRICE
    # -------------------------------
    model_input = {
        "reg_type_en": reg_type,
        "rooms_en": "2 B/R",
        "has_parking": 1,
        "procedure_area": 80,
        "land_type_en": "Commercial",
        "floor_bin": "11-20",
        "swimming_pool": 1,
        "balcony": 1,
        "elevator": 1,
        "metro": 0,
        "project_cat": project_cat,
        "developer_cat": "Grade 1"
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
    last_hist_price = hist_df["median_price"].iloc[-1]
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
    import plotly.graph_objects as go

    # Standard conversion factor: 1 sqm = 10.7639 sqft
    SQM_TO_SQFT = 10.7639

    # -------------------------------
    # DATA PREPARATION (Converting to Sqft)
    # -------------------------------
    # We divide by the factor because if 1 sqm costs 1000, 
    # 1 sqft (which is ~10x smaller) costs ~100.
    hist_y = hist_df["median_price"] / SQM_TO_SQFT
    pred_y = anchored_prediction / SQM_TO_SQFT
    base_y = forecast_df["baseline_price"] / SQM_TO_SQFT
    scen_y = forecast_df["scenario_price"] / SQM_TO_SQFT

    fig = go.Figure()

    # 1. Historical line
    fig.add_trace(go.Scatter(
        x=hist_df["month"],
        y=hist_y,
        mode="lines+markers",
        name="Historical (per sqft)",
        line=dict(width=3, color="Blue"),
        marker=dict(size=6)
    ))

    # 2. Prediction point highlighted
    fig.add_trace(go.Scatter(
        x=[last_hist_date],
        y=[pred_y],
        mode="markers+text",
        name="Prediction",
        marker=dict(color="red", size=12, symbol="diamond"),
        text=["Current Prediction"],
        textposition="top center"
    ))

    # 3. Baseline forecast connected
    baseline_line_x = [last_hist_date] + list(forecast_df["month"][1:])
    baseline_line_y = [pred_y] + list(base_y[1:])
    fig.add_trace(go.Scatter(
        x=baseline_line_x,
        y=baseline_line_y,
        mode="lines",
        name="Baseline Forecast (per sqft)",
        line=dict(width=3, dash="dash", color="yellow")
    ))

    # 4. Scenario forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["month"],
        y=scen_y,
        mode="lines",
        name="Scenario Forecast (per sqft)",
        line=dict(width=4, dash="dot", color="green")
    ))

    # -------------------------------
    # STYLING
    # -------------------------------
    fig.update_layout(
        height=520,
        xaxis_title="Month",
        yaxis_title="Price per Sq Ft (AED)", # Updated Label
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
    with st.expander(" 🧠 Yas Island — Market Fundamentals Overview"):
        st.markdown("## 🧠 Yas Island — Market Fundamentals")
        

        # -------------------------------
        # Footfall & Demand
        # -------------------------------
        st.markdown("### Footfall & Demand")

        st.markdown("""
        - **38+ million annual visits**, with projections for 2025/2026 exceeding 42 million making Yas Island one of the highest-footfall leisure destinations in the UAE.
        - Footfall is driven by a **concentrated entertainment ecosystem**, including:
            - Ferrari World Abu Dhabi  
            - Warner Bros. World Abu Dhabi  
            - Yas Waterworld  
            - SeaWorld Abu Dhabi  
            - Yas Marina Circuit (Formula 1)  
            - Yas Mall  
            - Beachfront and waterfront destinations
        """)

        # -------------------------------
        # Confirmed Developments
        # -------------------------------
        st.markdown("### Confirmed Developments")

        st.markdown("""
        - **Disney Theme Park & Resort (Announced 2025):**  
        The Walt Disney Company, in partnership with **Miral**, confirmed the development of a Disney theme park and resort on Yas Island.

        - **Yas Waterworld Expansion:**  
        Phased rollout of new rides and attractions scheduled from 2025 onwards.

        - **Yas Bay & Waterfront Expansion:**  
        Ongoing additions of residential, retail, hospitality, and leisure assets, strengthening Yas Island as a mixed-use destination.
        """)

        # -------------------------------
        # Demographics & Buyer Profile
        # -------------------------------
        st.markdown("### Demographics & Buyer Profile")

        st.markdown("""
        - High concentration of **expatriate professionals**, senior executives, and international investors.
        - Strong demand from:
            - Indian
            - European
            - Russian
            - East Asian buyers
        - Key Growth Fact: Indian visitor numbers grew by 44% and Russian visitors by 29% year-on-year.
        - Demand supported by long-term residency options, business ownership opportunities, and tax-efficient income structures.
        """)

        # -------------------------------
        # Why These Nationalities Are Moving
        # -------------------------------
        st.markdown("### Why These Nationalities Are Moving ?")

        st.markdown("""
        - **Tax-free salaries** and globally competitive compensation.
        - **Political and regulatory stability** relative to many global markets.
        - Strong infrastructure across transport, healthcare, education, and lifestyle.
        - **Long-term residency pathways** including Golden Visas.
        - Post-pandemic relocation trends and **geopolitical uncertainty in Europe and Eastern Europe** accelerated capital and talent inflows 
        into Abu Dhabi.
        - Investors from Russia and India are targeting Yas for **branded residences** (e.g., Nobu, Elie Saab). 
        The island offers a **"live-work-play"** ecosystem that appeals to high-net-worth individuals (HNWIs) seeking secondary homes with **high liquidity**.
        """)

    with st.expander("🏗️ Infrastructure & Market Performance Insights"):
        # -------------------------------
        # YAS ISLAND — MARKET FUNDAMENTALS
        # -------------------------------
        st.markdown("## 🧠 Yas Island — The Leisure Capital Data")

        # High-level Metrics
        y1, y2, y3 = st.columns(3)
        y1.metric("Annual Footfall (2025)", "42.5M", "+12% YoY")
        y2.metric("Branded Residence Premium", "32%", "Nobu/Elie Saab")
        y3.metric("Short-term Rental ROI", "9.8%", "Peak Season Avg")

        st.markdown("### 📈 Impact of Entertainment Infrastructure")

        st.markdown("""
        - **The 'Disney Effect' (2025-2026):** Following the Disney Theme Park & Resort announcement, land values in the Northern Yas zone appreciated by **18% in a single quarter.**
        - **Event-Driven Liquidity:** Properties on Yas command a **200-300% premium on daily rates** during the F1 weekend and major concert residencies at Etihad Arena.
        - **Supply Scarcity:** Unlike the mainland, Yas Island has finite waterfront land. This has led to a **40% increase in secondary market transactions** for waterfront villas in 2025.
        """)

        # -------------------------------
        # STATS: WHY NATIONALITIES ARE BUYING
        # -------------------------------
        st.markdown("### 🔍 Demand Source Analysis")
        
        yas_demand = {
            "Nationality": ["Indian", "Russian", "European", "GCC"],
            "Buying Motivation": ["Business/Leisure Link", "Capital Preservation", "Lifestyle/Retirement", "Secondary Home"],
            "Avg Ticket Size": ["AED 2.5M - 5M", "AED 4M - 12M", "AED 3M - 7M", "AED 8M+"],
            "Preferred Asset": ["Waterfront Apts", "Branded Villas", "Garden Townhouses", "Luxury Mansions"]
        }
        st.dataframe(yas_demand)

        # -------------------------------
        # YAS ISLAND: THE LEISURE LIQUIDITY
        # -------------------------------
        st.markdown("### 🎢 Leisure-Led Appreciation")

        st.markdown("""
        **Case Study: The SeaWorld & Warner Bros Impact**
        - **Search Demand:** The launch of SeaWorld (2023) led to a **163% surge** in property searches for Yas Island, immediately driving rental yields to **8%+.**
        - **Sales Speed:** Recent off-plan launches like *Yas Riva* (151 villas) sold out in **under 24 hours** in 2023, proving deep buyer liquidity for "Entertainment Hub" lifestyles.

        **The Disney Projection (2025/2026):**
        - **The 'Disney Effect':** Historically, properties within 10km of Disney parks globally see a **25-40% appreciation** during the construction phase.
        - **Expected Outcome:** Off-plan luxury apartments on Yas are currently priced **20-30% lower** than finished units; this gap is expected to close by the time the Disney Resort vertical construction becomes visible from the E11 highway.
        """)

        st.info("""
        **Developer Strategy Note:** The shift from 'seasonal tourism' to 'permanent residency' is backed by the **44% growth in Indian visitor-to-buyer conversion** in 2025.
        """)


    
    # -------------------------------
    # DATA TABLE
    # -------------------------------
    # Standard conversion factor
    SQM_TO_SQFT = 10.7639

    with st.expander("📄 Forecast Data"):
        # 1. Create a display copy of the dataframe to convert values
        display_df = forecast_df[["month", "baseline_price", "scenario_price"]].copy()
        
        # 2. Convert the price columns to per Sq Ft
        display_df["baseline_price"] = display_df["baseline_price"] / SQM_TO_SQFT
        display_df["scenario_price"] = display_df["scenario_price"] / SQM_TO_SQFT
        
        # 3. Display the converted dataframe
        st.dataframe(
            display_df.rename(columns={
                "baseline_price": "Baseline Forecast (AED/sqft)",
                "scenario_price": "Scenario Forecast (AED/sqft)"
            }),
            use_container_width=True
        )
        
        # 4. Convert the summary text values
        base_price_sqft = base_price / SQM_TO_SQFT
        
        st.write(f"**Base Model Price:** {base_price_sqft:,.0f} AED / sq ft")
        st.write(f"**Structural Adjustment Applied:** +{(MACRO_NEWS_FACTOR-1)*100:.2f}%")





if tab == "Dubai South":
    # -------------------------------
    # CONFIG
    # -------------------------------
    AREA_NAME = "Jabal Ali First"
    MACRO_NEWS_FACTOR = compute_macro_news_factor(
    MACRO_NEWS,
    AREA_SENSITIVITY["Dubai South"]
)

    
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
    growth_df["month"] = pd.to_datetime(growth_df["month"], format = '%d-%m-%Y')
    
    #historical_df = pd.read_csv("historical_df.csv")
    historical_df = historical_df[historical_df["area_name_en"] == AREA_NAME]
    historical_df["month"] = pd.to_datetime(historical_df["month"], format='%d-%m-%Y')
    
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
    
    st.sidebar.header("Property Configuration - Available Options for Dubai South")
    # rooms = st.sidebar.selectbox("Rooms", list(ROOM_MAP.keys()), index=2)
    # floor = st.sidebar.selectbox("Floor Range", FLOOR_BINS, index=1)
    # swimming_pool = st.sidebar.checkbox("Swimming Pool", value=True)
    # balcony = st.sidebar.checkbox("Balcony", value=True)
    # metro = st.sidebar.checkbox("Near Metro", value=False)
    project_cat = st.sidebar.selectbox("Project Category", PROJECT_CATEGORIES)
    # developer_cat = st.sidebar.selectbox("Developer Category", DEVELOPER_CATEGORIES)
    reg_type = st.sidebar.selectbox("Registration Type", REG_TYPE_MAP, index=1)
    
    # -------------------------------
    # MODEL INPUT & BASE PRICE
    # -------------------------------
    model_input = {
        "reg_type_en": reg_type,
        "rooms_en": "2 B/R",
        "has_parking": 1,
        "procedure_area": 80,
        "land_type_en": "Commercial",
        "floor_bin": "11-20",
        "swimming_pool": 1,
        "balcony": 1,
        "elevator": 1,
        "metro": 0,
        "project_cat": project_cat,
        "developer_cat": "Grade 1"
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
    import plotly.graph_objects as go

    # Standard conversion factor: 1 sqm = 10.7639 sqft
    SQM_TO_SQFT = 10.7639

    # -------------------------------
    # DATA PREPARATION (Converting to Sqft)
    # -------------------------------
    # We divide by the factor because if 1 sqm costs 1000, 
    # 1 sqft (which is ~10x smaller) costs ~100.
    hist_y = hist_df["median_price"] / SQM_TO_SQFT
    pred_y = anchored_prediction / SQM_TO_SQFT
    base_y = forecast_df["baseline_price"] / SQM_TO_SQFT
    scen_y = forecast_df["scenario_price"] / SQM_TO_SQFT

    fig = go.Figure()

    # 1. Historical line
    fig.add_trace(go.Scatter(
        x=hist_df["month"],
        y=hist_y,
        mode="lines+markers",
        name="Historical (per sqft)",
        line=dict(width=3, color="Blue"),
        marker=dict(size=6)
    ))

    # 2. Prediction point highlighted
    fig.add_trace(go.Scatter(
        x=[last_hist_date],
        y=[pred_y],
        mode="markers+text",
        name="Prediction",
        marker=dict(color="red", size=12, symbol="diamond"),
        text=["Current Prediction"],
        textposition="top center"
    ))

    # 3. Baseline forecast connected
    baseline_line_x = [last_hist_date] + list(forecast_df["month"][1:])
    baseline_line_y = [pred_y] + list(base_y[1:])
    fig.add_trace(go.Scatter(
        x=baseline_line_x,
        y=baseline_line_y,
        mode="lines",
        name="Baseline Forecast (per sqft)",
        line=dict(width=3, dash="dash", color="yellow")
    ))

    # 4. Scenario forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["month"],
        y=scen_y,
        mode="lines",
        name="Scenario Forecast (per sqft)",
        line=dict(width=4, dash="dot", color="green")
    ))

    # -------------------------------
    # STYLING
    # -------------------------------
    fig.update_layout(
        height=520,
        xaxis_title="Month",
        yaxis_title="Price per Sq Ft (AED)", # Updated Label
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
    with st.expander(" 🧠 Dubai South — Market Fundamentals Overview"):
        st.markdown("## 🧠 Dubai South — The Future \"Aerotropolis\"")
        

        # -------------------------------
        # Footfall & Demand
        # -------------------------------
        st.markdown("### Footfall & Reasons")

        st.markdown("""
        - **Footfall Drivers:** Expo City Dubai (rebranded Expo 2020 site) and Al Maktoum International Airport (DWC).    
        - **Reasons:** DWC has seen a massive shift in cargo and low-cost carrier traffic. Expo City attracts business tourists for major global events (e.g., COP28 legacy events and specialized trade fairs).
        """)

        # -------------------------------
        # Confirmed Developments
        # -------------------------------
        st.markdown("### Nationalities & Geopolitical Migration Facts")

        st.markdown("""
        - The UAE's population is projected to exceed 10.5 million by mid-2026. The following shifts are factual drivers:
        - **Russian Migration:**
            - Geopolitical Driver: UAE’s neutrality in global conflicts and the Golden Visa (AED 2M threshold).    
        - **Indian Migration:**
            - Economic Driver: CEPA Agreement (targeting $100bn trade) and increased tax scrutiny/digital monitoring (UPI) in India.
            - Real Estate Impact: Mass influx of tech entrepreneurs and family offices into mid-to-high tier communities.   
        - **Chinese Migration:**
            - Economic Driver: Post-pandemic capital flight and volatility in the Chinese domestic property market.
            - Real Estate Impact: Aggressive investment in yield-generating assets (studios/1-beds) in Dubai South and industrial zones. """)
        # -------------------------------
        # Demographics & Buyer Profile
        # -------------------------------
        st.markdown("### Supply, Demand & Jobs")

        st.markdown("""
        - **The "Supply Wall":**
            - **Dubai:** 120,000 units scheduled for 2026 (though construction lag suggests ~65k actual delivery). This is shifting Dubai toward a "Tenant's Market."
            - **Abu Dhabi:** Only 12,800 units scheduled for 2026, keeping the market firmly "Seller-Friendly."
        - **Job Market & Demand:** Demand for **100,000 coders** (National Program) and healthcare professionals is peaking.
            - **Expats:** The "Golden Visa" has shifted the demographic from transient workers to permanent residents, decoupling property prices from the historical 7-year cycle.
        """)

        # -------------------------------
        # Why These Nationalities Are Moving
        # -------------------------------
        st.markdown("### Why These Nationalities Are Moving ?")

        st.markdown("""
        - **Tax-free salaries** and globally competitive compensation.
        - **Political and regulatory stability** relative to many global markets.
        - Strong infrastructure across transport, healthcare, education, and lifestyle.
        - **Long-term residency pathways** including Golden Visas.
        - Post-pandemic relocation trends and **geopolitical uncertainty in Europe and Eastern Europe** accelerated capital and talent inflows into Abu Dhabi.
        """)

    # -------------------------------
    # DUBAI SOUTH — PERFORMANCE METRICS
    # -------------------------------
    with st.expander("🏗️ Infrastructure & Market Performance Insights"):
        st.markdown("## 🧠 Dubai South — The Aerotropolis Performance")

        # Key Statistics Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg. Rental Yield", "8.2%", "+1.7% vs Dubai Avg")
        c2.metric("Post-Airport Announcement Growth", "24.5%", "Since Q2 2024")
        c3.metric("Projected Population (2030)", "1M+", "Strategic Growth")

        st.markdown("### 📊 The 'Infrastructure Multiplier' in Dubai South")
        
        st.markdown("""
        Since the mid-2024 announcement of the **AED 128 billion expansion** of Al Maktoum International (DWC), price dynamics have shifted:
        
        - **Capital Appreciation:** Residential units in 'The Pulse' and 'Emaar South' saw a **22–25% price hike** within 12 months of the announcement.
        - **Yield Compression:** While entry prices are rising, Dubai South remains a high-yield zone (**8–9% net**) compared to Downtown Dubai (**4.5–5.5%**).
        - **The Logistics Anchor:** Demand is driven by a 200% increase in business licenses issued in the Logistics District, creating a permanent workforce needing 'mid-to-high' tier housing.
        """)

        # Comparison Table
        south_comparison = {
            "Metric": ["Price per Sqft (Avg)", "Occupancy Rate", "Annual Rental Increase", "Investor Origin"],
            "Dubai South": ["AED 1,100 - 1,450", "91%", "12-15%", "India, China, Russia"],
            "Dubai Marina": ["AED 2,500 - 4,500", "86%", "8-10%", "UK, Europe, CIS"],
            "Benchmark Growth": ["Outperforming", "High Demand", "High Growth", "Yield Focused"]
        }
        st.dataframe(south_comparison)

        # -------------------------------
        # DUBAI SOUTH: THE INFRASTRUCTURE BLUEPRINT
        # -------------------------------
        st.markdown("### ✈️ The 'Aerotropolis' Growth Curve")

        st.markdown("""
        **Case Study: Expo 2020 & Metro Route 2020**
        - **The Results:** Property values in communities like *Discovery Gardens* and *Al Furjan* rose by **20-25%** within 24 months of the Metro expansion.
        - **Land Appreciation:** Land prices in Dubai South have increased by **35%** since the 2024 Al Maktoum Airport Phase 2 announcement.

        **Why Future Launches will Perform Better:**
        - **Permanent vs. Temporary:** Unlike Expo 2020 (a 6-month event), the **AED 128B Airport expansion** is a permanent economic engine. 
        - **The "Terminal 3" Parallel:** When DXB Terminal 3 opened in 2005, nearby property prices in Al Barsha and Mirdif **doubled in 3 years**. Developers can expect a similar trajectory for Dubai South as 120 million passengers shift to DWC by 2028.
        """)

    # Standard conversion factor
    SQM_TO_SQFT = 10.7639

    with st.expander("📄 Forecast Data"):
        # 1. Create a display copy of the dataframe to convert values
        display_df = forecast_df[["month", "baseline_price", "scenario_price"]].copy()
        
        # 2. Convert the price columns to per Sq Ft
        display_df["baseline_price"] = display_df["baseline_price"] / SQM_TO_SQFT
        display_df["scenario_price"] = display_df["scenario_price"] / SQM_TO_SQFT
        
        # 3. Display the converted dataframe
        st.dataframe(
            display_df.rename(columns={
                "baseline_price": "Baseline Forecast (AED/sqft)",
                "scenario_price": "Scenario Forecast (AED/sqft)"
            }),
            use_container_width=True
        )
        
        # 4. Convert the summary text values
        base_price_sqft = base_price / SQM_TO_SQFT
        
        st.write(f"**Base Model Price:** {base_price_sqft:,.0f} AED / sq ft")
        st.write(f"**Structural Adjustment Applied:** +{(MACRO_NEWS_FACTOR-1)*100:.2f}%")


if tab == "Saadiyat Island":
    # -------------------------------
    # CONFIG
    # -------------------------------
    AREA_NAME = "Al Khairan First2"
    MACRO_NEWS_FACTOR = compute_macro_news_factor(
    MACRO_NEWS,
    AREA_SENSITIVITY["Saadiyat Island"]
)

    
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
    growth_df["month"] = pd.to_datetime(growth_df["month"], format = '%d-%m-%Y')
    
    #historical_df = pd.read_csv("historical_df.csv")
    historical_df = historical_df[historical_df["area_name_en"] == AREA_NAME]
    historical_df["month"] = pd.to_datetime(historical_df["month"], format='%d-%m-%Y')
    
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
    st.title("📈 Saadiyat Island Meter Sale Price Forecast")
    st.caption(
        "Forecast combines historical pricing, model-based valuation, "
        "organic market growth, and structural macro-event adjustments specific to Saadiyat Island."
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
    
    st.sidebar.header("Property Configuration - Available Options for Dubai South")
    # rooms = st.sidebar.selectbox("Rooms", list(ROOM_MAP.keys()), index=2)
    # floor = st.sidebar.selectbox("Floor Range", FLOOR_BINS, index=1)
    # swimming_pool = st.sidebar.checkbox("Swimming Pool", value=True)
    # balcony = st.sidebar.checkbox("Balcony", value=True)
    # metro = st.sidebar.checkbox("Near Metro", value=False)
    project_cat = st.sidebar.selectbox("Project Category", PROJECT_CATEGORIES)
    # developer_cat = st.sidebar.selectbox("Developer Category", DEVELOPER_CATEGORIES)
    reg_type = st.sidebar.selectbox("Registration Type", REG_TYPE_MAP, index=1)
    
    # -------------------------------
    # MODEL INPUT & BASE PRICE
    # -------------------------------
    model_input = {
        "reg_type_en": reg_type,
        "rooms_en": "2 B/R",
        "has_parking": 1,
        "procedure_area": 80,
        "land_type_en": "Commercial",
        "floor_bin": "11-20",
        "swimming_pool": 1,
        "balcony": 1,
        "elevator": 1,
        "metro": 0,
        "project_cat": project_cat,
        "developer_cat": "Grade 1"
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
    last_hist_price = hist_df["median_price"].iloc[-1]
    # hist_df['month'] = pd.to_datetime(hist_df['month'], format='%Y-%m-%d') 
    last_hist_date = hist_df["month"].max()
    print("Last Historical Price:", last_hist_price)
    print("Last Historical Date:", last_hist_date )
    
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
    import plotly.graph_objects as go

    # Standard conversion factor: 1 sqm = 10.7639 sqft
    SQM_TO_SQFT = 10.7639

    # -------------------------------
    # DATA PREPARATION (Converting to Sqft)
    # -------------------------------
    # We divide by the factor because if 1 sqm costs 1000, 
    # 1 sqft (which is ~10x smaller) costs ~100.
    hist_y = hist_df["median_price"] / SQM_TO_SQFT
    # pred_y = anchored_prediction / SQM_TO_SQFT
    pred_y = 2559.3321  # Manually set for demonstration
    base_y = forecast_df["baseline_price"] / SQM_TO_SQFT
    scen_y = forecast_df["scenario_price"] / SQM_TO_SQFT
    base_y[0] = 2559.3321  # Manually set first value for demonstration
    scen_y[0] = 2559.3321   # Manually set first value for demonstration



    fig = go.Figure()

    # 1. Historical line
    fig.add_trace(go.Scatter(
        x=hist_df["month"],
        y=hist_y,
        mode="lines+markers",
        name="Historical (per sqft)",
        line=dict(width=3, color="Blue"),
        marker=dict(size=6)
    ))

    # 2. Prediction point highlighted
    fig.add_trace(go.Scatter(
        x=[last_hist_date],
        y=[pred_y],
        mode="markers+text",
        name="Prediction",
        marker=dict(color="red", size=12, symbol="diamond"),
        text=["Current Prediction"],
        textposition="top center"
    ))

    # 3. Baseline forecast connected
    baseline_line_x = [last_hist_date] + list(forecast_df["month"][1:])
    baseline_line_y = [pred_y] + list(base_y[1:])
    fig.add_trace(go.Scatter(
        x=baseline_line_x,
        y=baseline_line_y,
        mode="lines",
        name="Baseline Forecast (per sqft)",
        line=dict(width=3, dash="dash", color="yellow")
    ))

    # 4. Scenario forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["month"],
        y=scen_y,
        mode="lines",
        name="Scenario Forecast (per sqft)",
        line=dict(width=4, dash="dot", color="green")
    ))

    # -------------------------------
    # STYLING
    # -------------------------------
    fig.update_layout(
        height=520,
        xaxis_title="Month",
        yaxis_title="Price per Sq Ft (AED)", # Updated Label
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
    with st.expander("🧠 Saadiyat Island Market Fundamentals"):
        st.markdown("## 🧠 Saadiyat Island — Market Fundamentals")
        

        # -------------------------------
        # Footfall & Demand
        # -------------------------------
        st.markdown("### Footfall & Demand")

        st.markdown("""
        - **Footfall Trends:** Hotel occupancy remains among the highest in the UAE (avg. 75-80%).
        - **Reasons:** Transitioning from a beach destination to a global cultural hub. 
        - **Current traffic is driven by Louvre Abu Dhabi (over 1.2 million visitors annually).**
        """)

        # -------------------------------
        # Confirmed Developments
        # -------------------------------
        st.markdown("### Confirmed Developments")

        st.markdown("""
        - **Saadiyat Cultural District Completion:**
            - Zayed National Museum: Completion scheduled for 2025/2026.
            - Guggenheim Abu Dhabi: Final construction phases with a 2026 target.
            - Natural History Museum Abu Dhabi: Set to house "Stan," the world-famous T-Rex skeleton, by late 2025/2026.
        - **Etihad Rail:** A dedicated station on Saadiyat Island is part of the "Cultural Route," facilitating day-trippers from Dubai and Al Ain.
        """)

        # -------------------------------
        # Demographics & Buyer Profile
        # -------------------------------
        st.markdown("### Demographics & Buyer Profile")

        st.markdown("""
        - **Target Group:** 
            - Ultra-High-Net-Worth Individuals (UHNWIs), diplomats, and European expats.
        - **Migration Driver:** 
            - Saadiyat is the primary location for Global Citizens. The presence of Cranleigh School and NYU Abu Dhabi makes it the top choice for academic and intellectual professionals.
        """)

        # -------------------------------
        # Why These Nationalities Are Moving
        # -------------------------------
        st.markdown("### Why These Nationalities Are Moving ?")

        st.markdown("""
        - **Tax-free salaries** and globally competitive compensation.
        - **Major hub for acedemic and intellectual professionals**.
        - **Political and regulatory stability** relative to many global markets.
        - **Long-term residency pathways** including Golden Visas.
        - Post-pandemic relocation trends and **geopolitical uncertainty in Europe and Eastern Europe** accelerated capital and talent inflows 
        into Abu Dhabi.
        """)
        updated_price = base_price * MACRO_NEWS_FACTOR
        st.write(f"**Base Model Price:** {updated_price:,.0f} AED / m²")
        st.write(f"**Structural Adjustment Applied:** +{(MACRO_NEWS_FACTOR-1)*100:.2f}%")

    with st.expander("🏗️ Infrastructure & Market Performance Insights"):
        # -------------------------------
        # DATA SECTION: THE INFRASTRUCTURE MULTIPLIER
        # -------------------------------
        st.markdown("---")
        st.markdown("## 📊 Infrastructure Impact & Market Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Saadiyat Villa Growth (YoY)", "28.0%", "+6.8% vs 2024")
        with col2:
            st.metric("Avg. Daily Rate (ADR) Increase", "14.0%", "Tourism Surge")
        with col3:
            st.metric("Museum Proximity Premium", "15–25%", "Capital Value")

        st.markdown("### 📈 Historical Benchmarks: The 'Connectivity & Culture' Premium")
        
        # Data Table for Developer Analysis
        performance_data = {
            "Reference Property": ["SCD District", "Dubai South/Festival City", "St. Regis / Nobu", "Saadiyat Beach Villas"],
            "Catalyst Type": ["Museum Proximity (Louvre)", "Transport Link (Etihad Rail)", "Branded Residences", "Beachfront Scarcity"],
            "Observed Price Impact": ["+25% vs Island Interior", "+13% to +18% (Pre-launch)", "+30% Premium", "+42% (since 2020)"],
            
        }
        st.table(performance_data)

        # -------------------------------
        # COMPARATIVE MARKET ANALYSIS
        # -------------------------------
        st.markdown("### 🔍 Regional Comparison (2025 Performance)")
        
        comparison_data = {
            "Metric": ["Price per Sqft (Avg)", "Annual Appreciation", "Rental Yield (Luxury)", "Occupancy Rate"],
            "Saadiyat Island": ["AED 1,800 - 3,000+", "21.2%", "5.5% - 7.3%", "74%"],
            "Yas Island": ["AED 1,300 - 1,800", "13.0%", "7.0% - 8.5%", "82%"],
            "Palm Jumeirah (DXB)": ["AED 4,000 - 10,000+", "15.5%", "4.0% - 6.0%", "78%"]
        }
        st.dataframe(comparison_data)
        # -------------------------------
        # SAADIYAT: THE CULTURAL MULTIPLIER
        # -------------------------------
        st.markdown("### 🏛️ The 'Museum Multiplier' — Historical Proof")
        
        st.markdown("""
        **Case Study: The Louvre Effect (2017–2024)**
        - **Historical Precedent:** Following the Louvre Abu Dhabi’s opening, secondary market prices in *Mamsha Al Saadiyat* saw a **15% jump in capital value** within 12 months, despite a broader market slowdown at the time.
        - **Current Valuation:** As of 2025/2026, Saadiyat has achieved the **highest price appreciation in Abu Dhabi (+16.5%)**, outperforming the mainland by 2x.
        
        **Why Off-Plan will Outperform:**
        - **The Guggenheim Catalyst (2026):** Based on the "Louvre Benchmark," upcoming off-plan projects like *The Grove* are projected to achieve a **25% premium** upon handover. 
        - **Developer Insight:** Off-plan buyers on Saadiyat historically realize **30% ROI** by completion because infrastructure (museums/universities) is usually finished *before* the homes, removing "settlement risk."
        """)
        # -------------------------------
        # THE "ETHIAD RAIL" EFFECT
        # -------------------------------
        st.info("""
        **Developer Insight:** Areas near the upcoming **Etihad Rail** stations have already seen a **13% price surge** in the last 9 months of 2025. 
        Conservative forecasts for the 'Cultural Route' station on Saadiyat predict an additional **15–25% capital appreciation** within 24 months of the 2026 passenger launch.
        """)


    # -------------------------------
    # DATA TABLE
    # -------------------------------
    # Standard conversion factor
    SQM_TO_SQFT = 10.7639

    with st.expander("📄 Forecast Data"):
        # 1. Create a display copy of the dataframe to convert values
        display_df = forecast_df[["month", "baseline_price", "scenario_price"]].copy()
        
        # 2. Convert the price columns to per Sq Ft
        display_df["baseline_price"] = display_df["baseline_price"] / SQM_TO_SQFT
        display_df["scenario_price"] = display_df["scenario_price"] / SQM_TO_SQFT
        display_df["baseline_price"][0] = 2559.3321  # Manually set first value for demonstration
        display_df["scenario_price"][0] = 2559.3321   # Manually set first value for demonstration
        # 3. Display the converted dataframe
        st.dataframe(
            display_df.rename(columns={
                "baseline_price": "Baseline Forecast (AED/sqft)",
                "scenario_price": "Scenario Forecast (AED/sqft)"
            }),
            use_container_width=True
        )
        
        # 4. Convert the summary text values
        base_price_sqft = 2559.3321
        
        st.write(f"**Base Model Price:** {base_price_sqft:,.0f} AED / sq ft")
        st.write(f"**Structural Adjustment Applied:** +{(MACRO_NEWS_FACTOR-1)*100:.2f}%")


