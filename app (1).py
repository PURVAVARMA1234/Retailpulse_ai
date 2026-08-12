import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

from sklearn.linear_model import LinearRegression


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RetailPulse AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto"
)


# ============================================================
# RESPONSIVE / THEME CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   GENERAL
============================================================ */

h1, h2, h3, h4, h5, h6 {
    color: var(--text-color) !important;
}

p, label, span {
    color: var(--text-color);
}


/* ============================================================
   SIDEBAR
   Native Streamlit sidebar only
============================================================ */

[data-testid="stSidebar"] {
    background-color: var(--secondary-background-color) !important;
}

[data-testid="stSidebar"] * {
    box-sizing: border-box;
}


/* ============================================================
   KPI CARDS
============================================================ */

[data-testid="stMetric"] {
    background-color: var(--secondary-background-color) !important;

    border: 1px solid rgba(128,128,128,0.25);

    border-radius: 16px;

    padding: 18px 20px;

    min-height: 120px;

    box-shadow: 0 4px 14px rgba(0,0,0,0.08);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);

    box-shadow:
        0 8px 22px rgba(0,0,0,0.13);
}

[data-testid="stMetricLabel"] {
    color: var(--text-color) !important;

    font-size: 14px !important;

    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-color) !important;

    font-size: 27px !important;

    font-weight: 700 !important;
}


/* ============================================================
   BUTTONS
============================================================ */

.stButton > button {
    border-radius: 10px;

    font-weight: 600;

    min-height: 42px;
}


/* ============================================================
   EXPANDER
============================================================ */

[data-testid="stExpander"] {
    border-radius: 14px;
}


/* ============================================================
   TOP NAVIGATION
============================================================ */

.rp-top-title {
    font-size: 20px;
    font-weight: 750;
    padding-top: 7px;
}


/* ============================================================
   GEMINI NAVIGATION LINK
============================================================ */

.rp-gemini-nav {
    display: inline-block;

    text-decoration: none !important;

    color: var(--text-color) !important;

    background: var(--secondary-background-color);

    border: 1px solid rgba(128,128,128,0.30);

    border-radius: 10px;

    padding: 8px 14px;

    font-weight: 650;

    transition: background 0.2s ease,
                transform 0.2s ease;
}

.rp-gemini-nav:hover {
    background: rgba(128,128,128,0.14);

    transform: translateY(-1px);
}


/* ============================================================
   GEMINI SECTION
============================================================ */

.rp-section {
    scroll-margin-top: 30px;
}


/* ============================================================
   MOBILE
============================================================ */

@media (max-width: 768px) {

    .block-container {
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }

    .rp-top-title {
        font-size: 16px;
    }

    .rp-gemini-nav {
        font-size: 13px;
        padding: 7px 10px;
    }

    [data-testid="stMetric"] {
        min-height: 100px;
        padding: 14px;
    }

    [data-testid="stMetricValue"] {
        font-size: 21px !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "combined_retail_data_clean.csv"
    )

    data["order_date"] = pd.to_datetime(
        data["order_date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["order_date"]
    ).copy()

    for col in [
        "sales",
        "profit",
        "quantity"
    ]:

        if col in data.columns:

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            ).fillna(0)

    return data


df = load_data()


# ============================================================
# PLOTLY THEME
# ============================================================

def get_plotly_template():

    try:

        if st.context.theme.type == "dark":
            return "plotly_dark"

        return "plotly_white"

    except Exception:

        return "plotly_white"


plot_template = get_plotly_template()


# ============================================================
# FORECAST FUNCTION
# ============================================================

def create_forecast(data):

    monthly_sales = (
        data
        .groupby(
            data["order_date"].dt.to_period("M")
        )["sales"]
        .sum()
        .reset_index()
    )

    monthly_sales["order_date"] = (
        monthly_sales["order_date"]
        .dt.to_timestamp()
    )

    monthly_sales = (
        monthly_sales
        .sort_values("order_date")
        .reset_index(drop=True)
    )

    forecast_dates = pd.date_range(
        start="2021-01-01",
        periods=12,
        freq="MS"
    )

    if monthly_sales.empty:

        forecast_values = np.zeros(12)

    elif len(monthly_sales) == 1:

        forecast_values = np.repeat(
            monthly_sales["sales"].iloc[0],
            12
        )

    else:

        monthly_sales["time_index"] = np.arange(
            len(monthly_sales)
        )

        X = monthly_sales[["time_index"]]

        y = monthly_sales["sales"]

        model = LinearRegression()

        model.fit(X, y)

        first_month = (
            monthly_sales["order_date"].min()
        )

        forecast_index = np.array([

            (
                (date.year - first_month.year) * 12
                +
                (date.month - first_month.month)
            )

            for date in forecast_dates

        ]).reshape(-1, 1)

        forecast_values = model.predict(
            forecast_index
        )

    forecast_values = np.maximum(
        forecast_values,
        0
    )

    forecast = pd.DataFrame({

        "Month":
            forecast_dates,

        "Predicted Sales":
            forecast_values

    })

    return monthly_sales, forecast


# ============================================================
# GEMINI BUSINESS CONTEXT
# ============================================================

def create_business_context(
    filtered_df,
    monthly_sales,
    forecast,
    selected_categories,
    selected_regions,
    selected_segments,
    start_date,
    end_date,
    dashboard_name
):

    category_sales = (
        filtered_df
        .groupby("category")["sales"]
        .sum()
    )

    category_profit = (
        filtered_df
        .groupby("category")["profit"]
        .sum()
    )

    region_sales = (
        filtered_df
        .groupby("region")["sales"]
        .sum()
    )

    region_profit = (
        filtered_df
        .groupby("region")["profit"]
        .sum()
    )

    context = f"""

RETAILPULSE AI
=============================

CURRENT DASHBOARD:
{dashboard_name}


SELECTED DATE RANGE:
{start_date.strftime("%d %b %Y")}
to
{end_date.strftime("%d %b %Y")}


TOTAL SALES:
{filtered_df["sales"].sum():.2f}


TOTAL PROFIT:
{filtered_df["profit"].sum():.2f}


TOTAL QUANTITY SOLD:
{filtered_df["quantity"].sum():.0f}


NUMBER OF RECORDS:
{len(filtered_df)}


SELECTED CATEGORIES:
{", ".join(map(str, selected_categories))}


SELECTED REGIONS:
{", ".join(map(str, selected_regions))}


SELECTED CUSTOMER SEGMENTS:
{", ".join(map(str, selected_segments))}


CATEGORY-WISE SALES:
{category_sales.to_string()}


CATEGORY-WISE PROFIT:
{category_profit.to_string()}


REGION-WISE SALES:
{region_sales.to_string()}


REGION-WISE PROFIT:
{region_profit.to_string()}


HISTORICAL MONTHLY SALES:
{
    monthly_sales[
        ["order_date", "sales"]
    ].to_string(index=False)
}


2021 SALES FORECAST:
{
    forecast.to_string(index=False)
}

"""

    return context


# ============================================================
# GEMINI API
# ============================================================

def ask_gemini(prompt):

    try:

        from google import genai

        api_key = None

        try:

            api_key = st.secrets.get(
                "GEMINI_API_KEY"
            )

        except Exception:

            pass

        if not api_key:

            api_key = os.environ.get(
                "GEMINI_API_KEY"
            )

        if not api_key:

            return (
                None,
                "Gemini API key is not configured. "
                "Please add GEMINI_API_KEY in Streamlit Secrets."
            )

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(

            model="gemini-3.5-flash-lite",

            contents=prompt
        )

        return response.text, None

    except Exception as e:

        return None, str(e)


# ============================================================
# GEMINI SECTION
# ============================================================

def show_gemini(
    business_context,
    dashboard_name,
    section_id
):

    st.markdown("---")

    # Navigation target
    st.markdown(
        f'<div id="{section_id}" class="rp-section"></div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "🤖 Gemini AI Business Assistant"
    )

    st.caption(
        f"AI assistant for {dashboard_name}. "
        "Use the standard analysis or ask your own question."
    )

    with st.expander(
        "🤖 Open Gemini AI",
        expanded=False
    ):

        # ====================================================
        # FIXED AI ANALYSIS
        # ====================================================

        fixed_prompt = f"""

You are the expert business analyst
inside RetailPulse AI.

Current dashboard:
{dashboard_name}


Analyze the following filtered retail data:

{business_context}


Give the answer in exactly these sections:


### 📈 TREND ANALYSIS

Analyze:

- Historical sales trends
- Profit trends
- Category performance
- Region performance
- Forecast trends


### 💡 BUSINESS INSIGHTS

Identify the most important
business findings.


### ❓ WHY IS THIS HAPPENING?

Explain possible reasons
using only the provided data.


### 🎯 RECOMMENDATIONS

Give exactly 5 practical
and actionable recommendations.


RULES:

- Do not invent numbers.
- Use only the provided data.
- Keep explanations simple.
- Focus on business decisions.
- Mention when the available data
  is insufficient.
"""

        if st.button(
            "✨ Generate AI Insights",
            key=f"generate_{section_id}"
        ):

            with st.spinner(
                "🤖 Gemini is analyzing..."
            ):

                result, error = ask_gemini(
                    fixed_prompt
                )

            if error:

                st.error(
                    f"❌ {error}"
                )

            else:

                st.success(
                    "✅ AI analysis generated successfully!"
                )

                st.markdown(result)


        # ====================================================
        # CUSTOM USER PROMPT
        # ====================================================

        st.markdown(
            "### 💬 Ask Gemini Anything"
        )

        user_prompt = st.text_area(

            "Enter your own business question",

            placeholder=(
                "Example: Which region needs the "
                "most attention and why?\n\n"
                "Example: How can I improve profit "
                "without reducing sales?"
            ),

            height=110,

            key=f"custom_prompt_{section_id}"
        )

        if st.button(
            "🚀 Ask Gemini",
            key=f"ask_{section_id}"
        ):

            if not user_prompt.strip():

                st.warning(
                    "Please enter your question first."
                )

            else:

                custom_prompt = f"""

You are Gemini,
the interactive business assistant
inside RetailPulse AI.


CURRENT DASHBOARD:
{dashboard_name}


USER QUESTION:
{user_prompt}


FILTERED BUSINESS DATA:
{business_context}


TASK:

Answer the user's question directly
using the supplied retail data.


RULES:

- Do not invent numbers.
- Use only the supplied data.
- If the data cannot answer something,
  clearly say so.
- Give practical business advice.
- Keep the answer simple and clear.
"""

                with st.spinner(
                    "🤖 Gemini is preparing your answer..."
                ):

                    result, error = ask_gemini(
                        custom_prompt
                    )

                if error:

                    st.error(
                        f"❌ {error}"
                    )

                else:

                    st.success(
                        "✅ Gemini response generated!"
                    )

                    st.markdown(result)


# ============================================================
# TOP NAVIGATION
# ============================================================

top_col1, top_col2 = st.columns(
    [5, 1],
    gap="small"
)

with top_col1:

    st.markdown(
        '<div class="rp-top-title">📊 RetailPulse AI</div>',
        unsafe_allow_html=True
    )

with top_col2:

    st.markdown(
        '[🤖 Gemini](#gemini-section)',
        unsafe_allow_html=False
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "RetailPulse AI"
)

st.markdown(
    """
    ### AI-Powered Retail Intelligence & Decision Support Platform

    Analyze business performance, explore customer and sales behavior,
    forecast future sales and get AI-driven business recommendations.
    """
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title(
    "🧭 Navigation"
)

dashboard = st.sidebar.radio(

    "Select Dashboard",

    [
        "📌 Executive Overview",
        "📈 Sales & Customer Analytics",
        "🔮 Forecast & AI"
    ],

    key="dashboard_navigation"
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🔎 Global Filters"
)


# CATEGORY

category_options = sorted(
    df["category"]
    .dropna()
    .unique()
)

selected_categories = st.sidebar.multiselect(

    "📦 Category",

    category_options,

    default=category_options,

    key="category_filter"
)


# REGION

region_options = sorted(
    df["region"]
    .dropna()
    .unique()
)

selected_regions = st.sidebar.multiselect(

    "🌍 Region",

    region_options,

    default=region_options,

    key="region_filter"
)


# CUSTOMER SEGMENT

segment_options = sorted(
    df["segment"]
    .dropna()
    .unique()
)

selected_segments = st.sidebar.multiselect(

    "👥 Customer Segment",

    segment_options,

    default=segment_options,

    key="segment_filter"
)


# ============================================================
# DATE RANGE
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📅 Date Range"
)

min_date = (
    df["order_date"]
    .min()
    .date()
)

max_date = (
    df["order_date"]
    .max()
    .date()
)

selected_date_range = st.sidebar.date_input(

    "Select Date Range",

    value=(
        min_date,
        max_date
    ),

    min_value=min_date,

    max_value=max_date,

    key="date_range_filter"
)


# ============================================================
# HANDLE DATE RANGE
# ============================================================

if isinstance(
    selected_date_range,
    (tuple, list)
):

    if len(selected_date_range) == 2:

        start_date = pd.to_datetime(
            selected_date_range[0]
        )

        end_date = pd.to_datetime(
            selected_date_range[1]
        )

    else:

        start_date = pd.to_datetime(
            selected_date_range[0]
        )

        end_date = start_date

else:

    start_date = pd.to_datetime(
        selected_date_range
    )

    end_date = start_date


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df[

    df["category"].isin(
        selected_categories
    )

    &

    df["region"].isin(
        selected_regions
    )

    &

    df["segment"].isin(
        selected_segments
    )

    &

    (df["order_date"] >= start_date)

    &

    (
        df["order_date"]
        <
        end_date + pd.Timedelta(days=1)
    )

].copy()


# ============================================================
# EMPTY DATA
# ============================================================

if filtered_df.empty:

    st.warning(
        "⚠️ No data available for "
        "the selected filters and date range."
    )

    st.stop()


# ============================================================
# FORECAST
# ============================================================

monthly_sales, forecast = create_forecast(
    filtered_df
)


# ============================================================
# GEMINI CONTEXT
# ============================================================

business_context = create_business_context(

    filtered_df,

    monthly_sales,

    forecast,

    selected_categories,

    selected_regions,

    selected_segments,

    start_date,

    end_date,

    dashboard
)


# ============================================================
# DASHBOARD 1
# EXECUTIVE OVERVIEW
# ============================================================

if dashboard == "📌 Executive Overview":

    st.header(
        "📌 Executive Overview"
    )

    st.caption(
        f"Selected period: "
        f"{start_date.strftime('%d %b %Y')} "
        f"to "
        f"{end_date.strftime('%d %b %Y')}"
    )


    # ========================================================
    # KPIs
    # ========================================================

    total_sales = filtered_df["sales"].sum()

    total_profit = filtered_df["profit"].sum()

    total_quantity = filtered_df["quantity"].sum()

    col1, col2, col3 = st.columns(
        3,
        gap="large"
    )

    with col1:

        st.metric(
            "💰 Total Sales",
            f"₹{total_sales:,.2f}"
        )

    with col2:

        st.metric(
            "📈 Total Profit",
            f"₹{total_profit:,.2f}"
        )

    with col3:

        st.metric(
            "📦 Quantity Sold",
            f"{total_quantity:,.0f}"
        )

    st.caption(
        f"Showing {len(filtered_df):,} records "
        f"out of {len(df):,} total records."
    )


    # ========================================================
    # CATEGORY + REGION
    # ========================================================

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        category_sales = (

            filtered_df
            .groupby("category")["sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_category = px.bar(

            category_sales,

            x="category",

            y="sales",

            title="📦 Sales by Category",

            labels={
                "category": "Category",
                "sales": "Sales"
            }
        )

        fig_category.update_layout(
            template=plot_template,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True
        )

    with col2:

        region_sales = (

            filtered_df
            .groupby("region")["sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_region = px.bar(

            region_sales,

            x="region",

            y="sales",

            title="🌍 Sales by Region",

            labels={
                "region": "Region",
                "sales": "Sales"
            }
        )

        fig_region.update_layout(
            template=plot_template,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_region,
            use_container_width=True
        )


    # ========================================================
    # HISTORICAL SALES
    # ========================================================

    st.subheader(
        "📈 Historical Monthly Sales"
    )

    fig_history = px.line(

        monthly_sales,

        x="order_date",

        y="sales",

        title="Historical Monthly Sales",

        labels={
            "order_date": "Month",
            "sales": "Sales"
        },

        markers=True
    )

    fig_history.update_layout(
        template=plot_template,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_history,
        use_container_width=True
    )


    # ========================================================
    # GEMINI
    # ========================================================

    show_gemini(
        business_context,
        dashboard,
        "gemini-section"
    )


# ============================================================
# DASHBOARD 2
# SALES & CUSTOMER ANALYTICS
# ============================================================

elif dashboard == "📈 Sales & Customer Analytics":

    st.header(
        "📈 Sales & Customer Analytics"
    )

    st.caption(
        "Understand category, customer "
        "and regional performance."
    )


    # ========================================================
    # KPIs
    # ========================================================

    col1, col2, col3 = st.columns(
        3,
        gap="large"
    )

    with col1:

        st.metric(
            "💰 Total Sales",
            f"₹{filtered_df['sales'].sum():,.2f}"
        )

    with col2:

        st.metric(
            "📈 Total Profit",
            f"₹{filtered_df['profit'].sum():,.2f}"
        )

    with col3:

        st.metric(
            "📦 Quantity Sold",
            f"{filtered_df['quantity'].sum():,.0f}"
        )


    # ========================================================
    # PROFIT + QUANTITY
    # ========================================================

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        profit_category = (

            filtered_df
            .groupby("category")["profit"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_profit = px.bar(

            profit_category,

            x="category",

            y="profit",

            title="💰 Profit by Category",

            labels={
                "category": "Category",
                "profit": "Profit"
            }
        )

        fig_profit.update_layout(
            template=plot_template,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_profit,
            use_container_width=True
        )

    with col2:

        quantity_category = (

            filtered_df
            .groupby("category")["quantity"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_quantity = px.bar(

            quantity_category,

            x="category",

            y="quantity",

            title="📦 Quantity Sold by Category",

            labels={
                "category": "Category",
                "quantity": "Quantity"
            }
        )

        fig_quantity.update_layout(
            template=plot_template,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_quantity,
            use_container_width=True
        )


    # ========================================================
    # SEGMENT + REGION
    # ========================================================

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        segment_sales = (

            filtered_df
            .groupby("segment")["sales"]
            .sum()
            .reset_index()
        )

        fig_segment = px.pie(

            segment_sales,

            names="segment",

            values="sales",

            hole=0.45,

            title="👥 Sales by Customer Segment"
        )

        fig_segment.update_layout(
            template=plot_template
        )

        st.plotly_chart(
            fig_segment,
            use_container_width=True
        )

    with col2:

        region_profit = (

            filtered_df
            .groupby("region")["profit"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_region_profit = px.bar(

            region_profit,

            x="region",

            y="profit",

            title="🌍 Profit by Region",

            labels={
                "region": "Region",
                "profit": "Profit"
            }
        )

        fig_region_profit.update_layout(
            template=plot_template,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_region_profit,
            use_container_width=True
        )


    # ========================================================
    # SALES TREND
    # ========================================================

    st.subheader(
        "📈 Monthly Sales Trend"
    )

    fig_sales_trend = px.line(

        monthly_sales,

        x="order_date",

        y="sales",

        title="Monthly Sales Trend",

        labels={
            "order_date": "Month",
            "sales": "Sales"
        },

        markers=True
    )

    fig_sales_trend.update_layout(
        template=plot_template,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_sales_trend,
        use_container_width=True
    )


    # ========================================================
    # GEMINI
    # ========================================================

    show_gemini(
        business_context,
        dashboard,
        "gemini-section"
    )


# ============================================================
# DASHBOARD 3
# FORECAST & AI
# ============================================================

else:

    st.header(
        "🔮 Forecast & AI"
    )

    st.caption(
        "Forecast future sales and "
        "get AI-powered business recommendations."
    )


    # ========================================================
    # KPIs
    # ========================================================

    col1, col2, col3 = st.columns(
        3,
        gap="large"
    )

    with col1:

        st.metric(
            "💰 Current Sales",
            f"₹{filtered_df['sales'].sum():,.2f}"
        )

    with col2:

        st.metric(
            "📈 Current Profit",
            f"₹{filtered_df['profit'].sum():,.2f}"
        )

    with col3:

        st.metric(
            "🔮 2021 Forecast",
            f"₹{forecast['Predicted Sales'].sum():,.2f}"
        )


    # ========================================================
    # HISTORICAL SALES
    # ========================================================

    st.subheader(
        "📈 Historical Monthly Sales"
    )

    fig_history = px.line(

        monthly_sales,

        x="order_date",

        y="sales",

        title="Historical Monthly Sales",

        labels={
            "order_date": "Month",
            "sales": "Sales"
        },

        markers=True
    )

    fig_history.update_layout(
        template=plot_template,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_history,
        use_container_width=True
    )


    # ========================================================
    # 2021 FORECAST
    # ========================================================

    st.subheader(
        "🔮 2021 Sales Forecast"
    )

    fig_forecast = px.line(

        forecast,

        x="Month",

        y="Predicted Sales",

        title="2021 Sales Forecast",

        labels={
            "Month": "Month",
            "Predicted Sales": "Predicted Sales"
        },

        markers=True
    )

    fig_forecast.update_layout(
        template=plot_template,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )


    # ========================================================
    # FORECAST TABLE
    # ========================================================

    st.subheader(
        "📋 Forecasted Sales"
    )

    display_forecast = forecast.copy()

    display_forecast["Month"] = (
        display_forecast["Month"]
        .dt.strftime("%b %Y")
    )

    display_forecast["Predicted Sales"] = (
        display_forecast["Predicted Sales"]
        .round(2)
    )

    st.dataframe(
        display_forecast,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # GEMINI
    # ========================================================

    show_gemini(
        business_context,
        dashboard,
        "gemini-section"
    )
