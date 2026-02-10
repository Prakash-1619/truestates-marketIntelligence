import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import plotly.graph_objects as go
import pydeck as pdk

# --- ENHANCED DATA REPOSITORY (Factual 2026 Revision) ---
area_data = {
    "Yas Island": {
        "vibe": "The Global Entertainment Metropolis",
        "yield": "8.5%", "yield_delta": "+1.2%",
        "jobs": "45,000+", "gdp_impact": "8.2%",
        "supply": 4500, "demand": 8200, "growth_score": 95,
        "expats": "Indian Tech HNWIs, UK 'Golden Visa' Retirees, Digital Nomads",
        "future_dev": "Disneyland Abu Dhabi (Phase 1), Harry Potter World, Etihad Rail Passenger Hub",
        "next_boom": "Yas North Branded Residences",
        "insight": "Property values near the Disney site have appreciated 18% since the 2025 announcement."
    },
    "Saadiyat Island": {
        "vibe": "The Starchitecture & Cultural Sanctuary",
        "yield": "7.2%", "yield_delta": "+0.8%",
        "jobs": "18,000+", "gdp_impact": "6.5%",
        "supply": 1200, "demand": 5800, "growth_score": 98,
        "expats": "European Diplomats, UHNWIs, Global Art Curators",
        "future_dev": "Guggenheim Abu Dhabi (Opening), Natural History Museum, Saadiyat Grove Retail",
        "next_boom": "Ramhan Island & Jubail Forest Villas",
        "insight": "Zero 'Beachfront' supply remains; secondary market premiums are at an all-time high."
    },
    "Dubai South": {
        "vibe": "The World's Aerotropolis",
        "yield": "10.5%", "yield_delta": "+2.1%",
        "jobs": "100,000+", "gdp_impact": "11.4%",
        "supply": 15000, "demand": 21000, "growth_score": 92,
        "expats": "Chinese Logistics Magnates, AI Engineers, Aviation Professionals",
        "future_dev": "DWC Airport (AED 128B Expansion), Metro Blue Line, Expo City Legacy Phase",
        "next_boom": "The 'South-West Corridor' (Dubai-Abu Dhabi Merger)",
        "insight": "Highest rental yields in the UAE driven by the relocation of major airline operations to DWC."
    }
}

def render_area(name):
    d = area_data[name]
    
    # 1. VISUAL HEADER & METRICS
    st.subheader(f"~ {d['vibe']}")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Net Rental Yield", d['yield'], d['yield_delta'])
    with m_col2:
        st.metric("Economic Jobs", d['jobs'], "Direct/Indirect")
    with m_col3:
        st.metric("District GVA", d['gdp_impact'], "of Non-Oil GDP")
    with m_col4:
        st.metric("Investment Grade", f"{d['growth_score']}/100", "Strong Buy")

    st.divider()

    # 2. DYNAMICS & ANALYSIS
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📊 Market Equilibrium (2026)")
        # Plotly Bar Chart with better styling
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Supply', 'Demand'], 
            y=[d['supply'], d['demand']],
            marker_color=['#00d4ff', '#ff4b4b'],
            text=[d['supply'], d['demand']],
            textposition='auto',
        ))
        fig.update_layout(
            height=300, 
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{name}_supply_demand")

    with col_right:
        st.info(f"**💡 Real-Time Insight:** {d['insight']}")
        st.markdown(f"""
        **🚀 The Next Boom:** {d['next_boom']}
        
        **🌍 Geopolitical Driver:** {d['expats']} are moving capital here due to the **UAE's Neutrality** and the **10-Year Golden Visa** which has now been lowered to a more accessible investment threshold.
        """)

    # 3. STRATEGIC INTELLIGENCE CARDS
    st.markdown("#### 🧠 Strategic Intelligence Brief")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 5px solid #00d4ff;">
            <strong>🏗️ Future Development</strong><br>
            {d['future_dev']}
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
            <strong>👥 Secondary Exit Profile</strong><br>
            Institutional REITs, Global Family Offices, and Indian 'New Wealth' seeking stability.
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 5px solid #00ff00;">
            <strong>📈 5-Year Outlook</strong><br>
            Full integration with <strong>Etihad Rail</strong> and {name} 2030 sustainability mandates.
        </div>
        """, unsafe_allow_html=True)

# Example execution (for the first tab)
# render_area("Yas Island")

# 1. Updated Data with 2026 Status Milestones
area_geo_data = {
    "yas_island": [
        {"name": "Disneyland Abu Dhabi", "lat": 24.490, "lon": 54.615, "intensity": 100, "status": "Site Confirmed (Jan 2026) / Construction Begun"},
        {"name": "Yas Waterworld Expansion", "lat": 24.487, "lon": 54.598, "intensity": 85, "status": "Opened 'Lost City' Expansion"},
        {"name": "Yas Bay Waterfront", "lat": 24.465, "lon": 54.600, "intensity": 90, "status": "2026 Luxury Handover Phase"},
        {"name": "Etihad Rail Yas Link", "lat": 24.502, "lon": 54.580, "intensity": 95, "status": "Passenger Ops Launching Q4 2026"}
    ],
    "saadiyat_island": [
        {"name": "Guggenheim Abu Dhabi", "lat": 24.535, "lon": 54.398, "intensity": 100, "status": "Final Fit-out / Opening Mid-2026"},
        {"name": "Zayed National Museum", "lat": 24.532, "lon": 54.405, "intensity": 95, "status": "Officially Opened (Feb 2026)"},
        {"name": "Natural History Museum", "lat": 24.538, "lon": 54.390, "intensity": 85, "status": "Opened Q4 2025"},
        {"name": "Saadiyat Grove", "lat": 24.530, "lon": 54.415, "intensity": 88, "status": "Retail & Residential Handover Peak"}
    ],
    "dubai_south": [
        {"name": "Al Maktoum Int. (DWC) Expansion", "lat": 24.900, "lon": 55.157, "intensity": 100, "status": "AED 128B Expansion - Phase 1 Active"},
        {"name": "Etihad Rail Dubai South Hub", "lat": 24.885, "lon": 55.130, "intensity": 98, "status": "Logistics & Passenger Terminal Active"},
        {"name": "Expo City Residential", "lat": 24.962, "lon": 55.150, "intensity": 92, "status": "D33 High-Growth Corridor"},
        {"name": "Metro Blue Line Link", "lat": 24.940, "lon": 55.170, "intensity": 90, "status": "Tunnelling Works Commenced"}
    ]
}

# 2. Dynamic Legends for each area
area_descriptions = {
    "yas_island": """
        * 🔴 **Disney North Zone:** Highest heat due to "The Disney Effect" on surrounding land.
        * 🟠 **Yas Bay:** 2026 peak liquidity for secondary market sales.
        * 🟡 **Waterworld/Entertainment:** Driving 22% footfall growth.
    """,
    "saadiyat_island": """
        * 🔴 **Cultural District:** World's highest concentration of "Starchitect" landmarks.
        * 🟠 **Saadiyat Grove:** High retail heat; the new center of island gravity.
        * 🟡 **Beachfront Scarcity:** 0% new supply left in prime coastal plots.
    """,
    "dubai_south": """
        * 🔴 **Aerotropolis Core:** Maximum heat around the world's largest airport project.
        * 🟠 **Logistics District:** High institutional interest and rental yield peak.
        * 🟡 **Expo City:** Sustained demand from the "15-minute city" lifestyle.
    """
}

def render_geo(area_id):
    # Ensure name matches keys
    df = pd.DataFrame(area_geo_data[area_id])
    title = area_id.replace('_', ' ').title()
    
    # --- FIX 1: DYNAMIC VIEWSTATE ---
    # Calculates the average Lat/Lon to center the map automatically
    mid_lat = df['lat'].mean()
    mid_lon = df['lon'].mean()

    with st.expander(f"📍 Geographic Expansion: The {title} Investment Map", expanded=True):
        st.write("### 🏗️ Project Catalysts & Growth Zones")
        
        layer = pdk.Layer(
            "HeatmapLayer",
            df,
            get_position=["lon", "lat"],
            get_weight="intensity",
            radius_pixels=80,
            intensity=1,
            threshold=0.05
        )
        
        # Centering the view on the specific area
        view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=11.5, pitch=45)
        
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            # map_style="mapbox://styles/mapbox/dark-v10",
            tooltip={"text": "{name}\nStatus: {status}"}
        ))
        
        # --- FIX 2: DYNAMIC LEGEND ---
        st.markdown("**Legend & Expansion Nodes:**")
        st.markdown(area_descriptions[area_id])

# Example Call:
# render_geo("yas_island")
# render_geo("saadiyat_island")
# render_geo("dubai_south")


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

# Tab 1

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
    st.title("📈 Yas Island")

    render_area("Yas Island")
    st.markdown("\n---\n")
    st.markdown("## 📊 Price Forecast")

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
    
    # st.sidebar.header("Property Configuration - Available Options for Dubai South")
    # # rooms = st.sidebar.selectbox("Rooms", list(ROOM_MAP.keys()), index=2)
    # # floor = st.sidebar.selectbox("Floor Range", FLOOR_BINS, index=1)
    # # swimming_pool = st.sidebar.checkbox("Swimming Pool", value=True)
    # # balcony = st.sidebar.checkbox("Balcony", value=True)
    # # metro = st.sidebar.checkbox("Near Metro", value=False)
    # project_cat = st.sidebar.selectbox("Project Category", PROJECT_CATEGORIES)
    # # developer_cat = st.sidebar.selectbox("Developer Category", DEVELOPER_CATEGORIES)
    # reg_type = st.sidebar.selectbox("Registration Type", REG_TYPE_MAP, index=1)
    
    # -------------------------------
    # MODEL INPUT & BASE PRICE
    # -------------------------------
    model_input = {
        "reg_type_en": "Existing Properties",
        "rooms_en": "2 B/R",
        "has_parking": 1,
        "procedure_area": 80,
        "land_type_en": "Commercial",
        "floor_bin": "11-20",
        "swimming_pool": 1,
        "balcony": 1,
        "elevator": 1,
        "metro": 0,
        "project_cat": "Standard Residential",
        "developer_cat": "Grade 1"
    }
    
    X = prepare_input(model_input)
    base_price = model.predict(X)[0]
    base_price = base_price * 1.03
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
    st.markdown("\n---\n")
    st.markdown("## 🧠 Market Fundamentals")
    

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

    st.write(f"**Base Model Price:** {base_price:,.0f} AED / m²")
    st.write(f"**Structural Adjustment Applied:** +{(MACRO_NEWS_FACTOR-1)*100:.2f}%")

    render_geo("yas_island")
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

    # with st.expander("Market Intelligence"):
    #     render_area("Yas Island")

    




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
    st.title("📈 Dubai South")
    render_area("Dubai South")
    st.markdown("\n---\n")
    st.markdown("## 📊 Price Forecast")

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
    
    # st.sidebar.header("Property Configuration - Available Options for Dubai South")
    # # rooms = st.sidebar.selectbox("Rooms", list(ROOM_MAP.keys()), index=2)
    # # floor = st.sidebar.selectbox("Floor Range", FLOOR_BINS, index=1)
    # # swimming_pool = st.sidebar.checkbox("Swimming Pool", value=True)
    # # balcony = st.sidebar.checkbox("Balcony", value=True)
    # # metro = st.sidebar.checkbox("Near Metro", value=False)
    # project_cat = st.sidebar.selectbox("Project Category", PROJECT_CATEGORIES)
    # # developer_cat = st.sidebar.selectbox("Developer Category", DEVELOPER_CATEGORIES)
    # reg_type = st.sidebar.selectbox("Registration Type", REG_TYPE_MAP, index=1)
    
    # -------------------------------
    # MODEL INPUT & BASE PRICE
    # -------------------------------
    model_input = {
        "reg_type_en": "Existing Properties",
        "rooms_en": "2 B/R",
        "has_parking": 1,
        "procedure_area": 80,
        "land_type_en": "Commercial",
        "floor_bin": "11-20",
        "swimming_pool": 1,
        "balcony": 1,
        "elevator": 1,
        "metro": 0,
        "project_cat": "Standard Residential",
        "developer_cat": "Grade 1"
    }
    
    X = prepare_input(model_input)
    base_price = model.predict(X)[0]
    base_price = base_price * 1.05
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

    st.write(f"**Base Model Price:** {base_price:,.0f} AED / m²")
    st.write(f"**Structural Adjustment Applied:** +{(MACRO_NEWS_FACTOR-1)*100:.2f}%")


    render_geo("dubai_south")

    with st.expander("📄 Forecast Data"):
        st.dataframe(
            forecast_df[["month", "baseline_price", "scenario_price"]]
            .rename(columns={
                "baseline_price": "Baseline Forecast",
                "scenario_price": "Scenario Forecast"
            })
        )


    



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
    st.title("📈 Saadiyat Island")

    render_area("Saadiyat Island")
    st.markdown("\n---\n")
    st.markdown("## 📊 Price Forecast")

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
    
    # st.sidebar.header("Property Configuration - Available Options for Dubai South")
    # # rooms = st.sidebar.selectbox("Rooms", list(ROOM_MAP.keys()), index=2)
    # # floor = st.sidebar.selectbox("Floor Range", FLOOR_BINS, index=1)
    # # swimming_pool = st.sidebar.checkbox("Swimming Pool", value=True)
    # # balcony = st.sidebar.checkbox("Balcony", value=True)
    # # metro = st.sidebar.checkbox("Near Metro", value=False)
    # project_cat = st.sidebar.selectbox("Project Category", PROJECT_CATEGORIES)
    # # developer_cat = st.sidebar.selectbox("Developer Category", DEVELOPER_CATEGORIES)
    # reg_type = st.sidebar.selectbox("Registration Type", REG_TYPE_MAP, index=1)
    
    # -------------------------------
    # MODEL INPUT & BASE PRICE
    # -------------------------------
    model_input = {
        "reg_type_en": "Existing Properties",
        "rooms_en": "2 B/R",
        "has_parking": 1,
        "procedure_area": 80,
        "land_type_en": "Commercial",
        "floor_bin": "11-20",
        "swimming_pool": 1,
        "balcony": 1,
        "elevator": 1,
        "metro": 0,
        "project_cat": "Standard Residential",
        "developer_cat": "Grade 1"
    }
    
    X = prepare_input(model_input)
    base_price = model.predict(X)[0]

    base_price = base_price * 1.09
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
    render_geo("saadiyat_island")

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



    



