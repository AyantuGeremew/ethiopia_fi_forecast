import streamlit as st

from data_loader import *
from metrics import *
from visualizations import *
from forecasts import *
from config import *


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide"
)


st.title(APP_TITLE)



# -----------------------------
# Load Data
# -----------------------------


df = load_data(
    "data/financial_inclusion.csv"
)


forecast_df = load_forecasts(
    "data/forecasts.csv"
)



scenario_df = load_scenarios(
    "data/scenarios.csv"
)



# -----------------------------
# Sidebar
# -----------------------------


page = st.sidebar.selectbox(
    "Navigation",
    [
        "Overview",
        "Trends",
        "Forecasts",
        "Inclusion Projections"
    ]
)



# ===================================================
# OVERVIEW PAGE
# ===================================================

if page=="Overview":


    st.header(
        "Financial Inclusion Overview"
    )


    col1,col2,col3,col4 = st.columns(4)


    current = current_value(
        df,
        "inclusion_rate"
    )


    growth = calculate_growth_rate(
        df,
        "inclusion_rate"
    )


    ratio = p2p_atm_ratio(df)



    col1.metric(
        "Current Inclusion Rate",
        f"{current}%"
    )


    col2.metric(
        "Growth Rate",
        f"{growth}%"
    )


    col3.metric(
        "P2P / ATM Ratio",
        ratio
    )


    col4.metric(
        "Records",
        len(df)
    )


    st.plotly_chart(
        time_series_plot(
            df,
            "inclusion_rate"
        ),
        use_container_width=True
    )



# ===================================================
# TRENDS PAGE
# ===================================================

elif page=="Trends":


    st.header(
        "Channel Trends Analysis"
    )


    start,end = st.date_input(
        "Select Date Range",
        [
            df.date.min(),
            df.date.max()
        ]
    )


    filtered=df[
        (df.date>=str(start))
        &
        (df.date<=str(end))
    ]


    st.plotly_chart(
        time_series_plot(
            filtered,
            "mobile_money"
        ),
        use_container_width=True
    )


    st.plotly_chart(
        channel_comparison(
            filtered
        ),
        use_container_width=True
    )


    csv = filtered.to_csv(index=False)


    st.download_button(
        "Download Trend Data",
        csv,
        "trend_data.csv"
    )



# ===================================================
# FORECAST PAGE
# ===================================================

elif page=="Forecasts":


    st.header(
        "Forecast Analysis"
    )


    model = st.selectbox(
        "Select Forecast Model",
        [
            "Linear Trend",
            "ARIMA",
            "Prophet"
        ]
    )


    selected = select_forecast_model(
        model,
        forecast_df
    )


    st.plotly_chart(
        forecast_plot(
            selected
        ),
        use_container_width=True
    )


    st.subheader(
        "Projected Milestones"
    )


    st.dataframe(
        selected[
            [
            "date",
            "forecast"
            ]
        ]
    )



# ===================================================
# INCLUSION PROJECTION PAGE
# ===================================================

elif page=="Inclusion Projections":


    st.header(
        "Financial Inclusion Projection Scenarios"
    )


    scenario = st.selectbox(
        "Scenario",
        SCENARIOS
    )


    selected=scenario_df[
        scenario_df.scenario==scenario
    ]



    current = selected.forecast.iloc[-1]



    st.plotly_chart(
        target_progress_chart(
            current,
            TARGET_RATE
        ),
        use_container_width=True
    )



    st.metric(
        "Projected Inclusion Rate",
        f"{current}%"
    )



    st.subheader(
        "Consortium Questions"
    )


    st.write(
    """
    ### 1. Will Ethiopia reach 60% inclusion target?

    Answer:
    Projection depends on scenario assumptions.
    Optimistic scenario assumes accelerated digital adoption.

    ---
    
    ### 2. Which channels drive growth?

    Mobile money and digital payments
    contribute the largest expected increase.

    ---
    
    ### 3. What risks can slow progress?

    - Digital literacy gaps
    - Network coverage limitations
    - Affordability constraints
    - Gender access gap

    ---
    
    ### 4. What interventions accelerate inclusion?

    - Agent expansion
    - Digital ID integration
    - Financial education
    - Affordable mobile services

    """
    )



    st.download_button(
        "Download Projection Data",
        selected.to_csv(index=False),
        "scenario_projection.csv"
    )