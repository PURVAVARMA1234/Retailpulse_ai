
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

from sklearn.linear_model import LinearRegression


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RetailPulse AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================
   GENERAL
   ========================= */

h1, h2, h3, h4 {
    color: var(--text-color) !important;
}

p, label, span {
    color: var(--text-color);
}


/* =========================
   SIDEBAR
   ========================= */

[data-testid="stSidebar"] {
    background-color: var(--secondary-background-color) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-color) !important;
}


/* =========================
   KPI CARDS
   ========================= */

[data-testid="stMetric"] {

    background-color:
        var(--secondary-background-color) !important;

    border: 1px solid
        rgba(128, 128, 128, 0.25);

    border-radius: 16px;

    padding: 20px 22px;

    min-height: 125px;

    box-shadow:
        0 4px 14px rgba(0, 0, 0, 0.08);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {

    transform: translateY(-3px);

    box-shadow:
        0 8px 22px rgba(0, 0, 0, 0.13);
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


/* =========================
   BUTTONS
   ========================= */

.stButton > button {

    border-radius: 10px;

    font-weight: 600;

    min-height: 42px;
}


/* =========================
   SELECTBOX / MULTISELECT
   ========================= */

[data-baseweb="select"] {

    border-radius: 8px;
}


/* =========================
   EXPANDER
   ========================= */

[data-testid="stExpander"] {

    border-radius: 14px;
}


/* =========================
   DIVIDER
   ========================= */

hr {

    opacity: 0.25;
}


/* =========================
   CUSTOM HEADER
   ========================= */

.retail-header {

    padding: 18px 0 8px 0;
}

.retail-subtitle {

    color: var(--text-color);

    opacity: 0.75;

    font-size: 16px;
}

.dashboard-badge {

    display: inline-block;

    padding: 6px 12px;

    border-radius: 20px;

    background: var(--secondary-background-color);

    border: 1px solid
        rgba(128,128,128,0.25);

    font-size: 13px;

    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

DATA_FILE = "combined_retail_data_clean.csv"


@st.cache_data
def load_data():

    data = pd.read_csv(DATA_FILE)

    data["order_date"] = pd.to_datetime(
        data["order_date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["order_date"]
    ).copy()

    return data


try:

    df = load_data()

except Exception as e:

    st.error(
        f"❌ Could not load {DATA_FILE}"
    )

    st.exception(e)

    st.stop()


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "order_date",
    "category",
    "region",
    "segment",
    "sales",
    "profit",
    "quantity"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "❌ Required columns are missing:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="retail-header">

    # 📊 RetailPulse AI

    <div class="retail-subtitle">
    AI-Powered Retail Intelligence & Decision Support Platform
    </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Analyze business performance, explore customer and sales behavior, "
    "forecast future sales and ask Gemini AI for business decisions."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 RetailPulse AI")

st.sidebar.markdown(
    "### 🧭 Navigation"
)

dashboard = st.sidebar.radio(
    "Select Dashboard",
    [
        "📌 Executive Overview",
        "📈 Sales & Customer Analytics",
        "🔮 Forecast & AI"
    ]
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 🔎 Global Filters"
)


# CATEGORY

category_options = sorted(
    df["category"]
    .dropna()
    .unique()
    .tolist()
)

selected_categories = st.sidebar.multiselect(
    "📦 Category",
    category_options,
    default=category_options
)


# REGION

region_options = sorted(
    df["region"]
    .dropna()
    .unique()
    .tolist()
)

selected_regions = st.sidebar.multiselect(
    "🌍 Region",
    region_options,
    default=region_options
)


# SEGMENT

segment_options = sorted(
    df["segment"]
    .dropna()
    .unique()
    .tolist()
)

selected_segments = st.sidebar.multiselect(
    "👥 Customer Segment",
    segment_options,
    default=segment_options
)


# ============================================================
# DATE RANGE
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 📅 Date Range"
)

min_date = df["order_date"].min().date()
max_date = df["order_date"].max().date()

selected_date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)


if isinstance(
    selected_date_range,
    tuple
) and len(selected_date_range) == 2:

    start_date = pd.Timestamp(
        selected_date_range[0]
    )

    end_date = pd.Timestamp(
        selected_date_range[1]
    )

else:

    start_date = pd.Timestamp(
        min_date
    )

    end_date = pd.Timestamp(
        max_date
    )


# ============================================================
# APPLY ALL FILTERS
# ============================================================

filtered_df = df[
    (
        df["category"]
        .isin(selected_categories)
    )
    &
    (
        df["region"]
        .isin(selected_regions)
    )
    &
    (
        df["segment"]
        .isin(selected_segments)
    )
    &
    (
        df["order_date"]
        .between(
            start_date,
            end_date + pd.Timedelta(days=1)
        )
    )
].copy()


# ============================================================
# EMPTY FILTER CHECK
# ============================================================

if filtered_df.empty:

    st.warning(
        "⚠️ No data available for the selected filters."
    )

    st.info(
        "Try selecting a wider date range or "
        "different filter values."
    )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def currency(value):

    return f"₹{value:,.2f}"


def chart_template():

    try:

        if st.context.theme.type == "dark":

            return "plotly_dark"

    except Exception:

        pass

    return "plotly_white"


def apply_chart_style(fig):

    fig.update_layout(
        template=chart_template(),
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


# ============================================================
# COMMON BUSINESS METRICS
# ============================================================

total_sales = filtered_df["sales"].sum()

total_profit = filtered_df["profit"].sum()

total_quantity = filtered_df["quantity"].sum()

profit_margin = (
    (total_profit / total_sales) * 100
    if total_sales != 0
    else 0
)

total_orders = (
    filtered_df["order_id"].nunique()
    if "order_id" in filtered_df.columns
    else len(filtered_df)
)

total_customers = (
    filtered_df["customer_id"].nunique()
    if "customer_id" in filtered_df.columns
    else 0
)


# ============================================================
# DASHBOARD 1
# EXECUTIVE OVERVIEW
# ============================================================

if dashboard == "📌 Executive Overview":

    st.markdown(
        '<span class="dashboard-badge">'
        'DASHBOARD 1 • EXECUTIVE OVERVIEW'
        '</span>',
        unsafe_allow_html=True
    )

    st.header(
        "📌 Executive Overview"
    )

    st.caption(
        "High-level view of the selected business performance."
    )


    # --------------------------------------------------------
    # KPI ROW 1
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "💰 Total Sales",
            currency(total_sales)
        )

    with col2:

        st.metric(
            "📈 Total Profit",
            currency(total_profit)
        )

    with col3:

        st.metric(
            "📦 Quantity Sold",
            f"{total_quantity:,.0f}"
        )

    with col4:

        st.metric(
            "📊 Profit Margin",
            f"{profit_margin:.2f}%"
        )


    # --------------------------------------------------------
    # KPI ROW 2
    # --------------------------------------------------------

    col5, col6, col7 = st.columns(3)

    with col5:

        st.metric(
            "🧾 Total Orders",
            f"{total_orders:,}"
        )

    with col6:

        st.metric(
            "👥 Customers",
            f"{total_customers:,}"
        )

    with col7:

        st.metric(
            "📋 Records",
            f"{len(filtered_df):,}"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # CATEGORY SALES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        category_sales = (
            filtered_df
            .groupby("category")["sales"]
            .sum()
            .sort_values(
                ascending=False
            )
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
            },
            text_auto=".2s"
        )

        apply_chart_style(
            fig_category
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True
        )


    # --------------------------------------------------------
    # REGION SALES
    # --------------------------------------------------------

    with col2:

        region_sales = (
            filtered_df
            .groupby("region")["sales"]
            .sum()
            .sort_values(
                ascending=False
            )
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
            },
            text_auto=".2s"
        )

        apply_chart_style(
            fig_region
        )

        st.plotly_chart(
            fig_region,
            use_container_width=True
        )


    # --------------------------------------------------------
    # SALES TREND
    # --------------------------------------------------------

    monthly_sales_exec = (
        filtered_df
        .groupby(
            filtered_df["order_date"]
            .dt.to_period("M")
        )["sales"]
        .sum()
        .reset_index()
    )

    monthly_sales_exec["order_date"] = (
        monthly_sales_exec["order_date"]
        .dt.to_timestamp()
    )

    fig_trend = px.line(
        monthly_sales_exec,
        x="order_date",
        y="sales",
        title="📈 Historical Monthly Sales",
        labels={
            "order_date": "Month",
            "sales": "Sales"
        },
        markers=True
    )

    apply_chart_style(
        fig_trend
    )

    st.plotly_chart(
        fig_trend,
        use_container_width=True
    )


# ============================================================
# DASHBOARD 2
# SALES & CUSTOMER ANALYTICS
# ============================================================

elif dashboard == "📈 Sales & Customer Analytics":

    st.markdown(
        '<span class="dashboard-badge">'
        'DASHBOARD 2 • SALES & CUSTOMER ANALYTICS'
        '</span>',
        unsafe_allow_html=True
    )

    st.header(
        "📈 Sales & Customer Analytics"
    )

    st.caption(
        "Understand category, customer and regional performance."
    )


    # --------------------------------------------------------
    # CATEGORY ANALYSIS
    # --------------------------------------------------------

    category_analysis = (
        filtered_df
        .groupby("category")
        .agg(
            Sales=("sales", "sum"),
            Profit=("profit", "sum"),
            Quantity=("quantity", "sum")
        )
        .reset_index()
        .sort_values(
            "Sales",
            ascending=False
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        fig_cat_profit = px.bar(
            category_analysis,
            x="category",
            y="Profit",
            title="💰 Profit by Category",
            labels={
                "category": "Category",
                "Profit": "Profit"
            },
            text_auto=".2s"
        )

        apply_chart_style(
            fig_cat_profit
        )

        st.plotly_chart(
            fig_cat_profit,
            use_container_width=True
        )


    with col2:

        fig_cat_quantity = px.bar(
            category_analysis,
            x="category",
            y="Quantity",
            title="📦 Quantity Sold by Category",
            labels={
                "category": "Category",
                "Quantity": "Quantity"
            },
            text_auto=".2s"
        )

        apply_chart_style(
            fig_cat_quantity
        )

        st.plotly_chart(
            fig_cat_quantity,
            use_container_width=True
        )


    st.markdown("---")


    # --------------------------------------------------------
    # CUSTOMER SEGMENT
    # --------------------------------------------------------

    if "segment" in filtered_df.columns:

        segment_sales = (
            filtered_df
            .groupby("segment")["sales"]
            .sum()
            .reset_index()
            .sort_values(
                "sales",
                ascending=False
            )
        )

        col1, col2 = st.columns(2)


        with col1:

            fig_segment = px.pie(
                segment_sales,
                names="segment",
                values="sales",
                hole=0.55,
                title="👥 Sales by Customer Segment"
            )

            apply_chart_style(
                fig_segment
            )

            st.plotly_chart(
                fig_segment,
                use_container_width=True
            )


        with col2:

            segment_profit = (
                filtered_df
                .groupby("segment")["profit"]
                .sum()
                .reset_index()
                .sort_values(
                    "profit",
                    ascending=False
                )
            )

            fig_segment_profit = px.bar(
                segment_profit,
                x="segment",
                y="profit",
                title="💰 Profit by Customer Segment",
                labels={
                    "segment": "Segment",
                    "profit": "Profit"
                },
                text_auto=".2s"
            )

            apply_chart_style(
                fig_segment_profit
            )

            st.plotly_chart(
                fig_segment_profit,
                use_container_width=True
            )


    st.markdown("---")


    # --------------------------------------------------------
    # TOP CUSTOMERS
    # --------------------------------------------------------

    if "customer_name" in filtered_df.columns:

        top_customers = (
            filtered_df
            .groupby("customer_name")["sales"]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(10)
            .reset_index()
        )

        fig_customers = px.bar(
            top_customers,
            x="sales",
            y="customer_name",
            orientation="h",
            title="🏆 Top 10 Customers by Sales",
            labels={
                "sales": "Sales",
                "customer_name": "Customer"
            },
            text_auto=".2s"
        )

        fig_customers.update_layout(
            yaxis=dict(
                categoryorder="total ascending"
            )
        )

        apply_chart_style(
            fig_customers
        )

        st.plotly_chart(
            fig_customers,
            use_container_width=True
        )


    # --------------------------------------------------------
    # REGION PROFITABILITY
    # --------------------------------------------------------

    region_analysis = (
        filtered_df
        .groupby("region")
        .agg(
            Sales=("sales", "sum"),
            Profit=("profit", "sum")
        )
        .reset_index()
    )

    fig_region_profit = px.bar(
        region_analysis,
        x="region",
        y=["Sales", "Profit"],
        barmode="group",
        title="🌍 Regional Sales vs Profit"
    )

    apply_chart_style(
        fig_region_profit
    )

    st.plotly_chart(
        fig_region_profit,
        use_container_width=True
    )


    # --------------------------------------------------------
    # ANALYTICS TABLE
    # --------------------------------------------------------

    st.subheader(
        "📋 Category Performance Summary"
    )

    display_category = category_analysis.copy()

    display_category["Sales"] = (
        display_category["Sales"]
        .round(2)
    )

    display_category["Profit"] = (
        display_category["Profit"]
        .round(2)
    )

    st.dataframe(
        display_category,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DASHBOARD 3
# FORECAST & AI
# ============================================================

elif dashboard == "🔮 Forecast & AI":

    st.markdown(
        '<span class="dashboard-badge">'
        'DASHBOARD 3 • FORECAST & AI'
        '</span>',
        unsafe_allow_html=True
    )

    st.header(
        "🔮 Forecast & AI Intelligence"
    )

    st.caption(
        "Forecast future sales and use Gemini AI "
        "for business decision support."
    )


    # ========================================================
    # HISTORICAL SALES
    # ========================================================

    monthly_sales = (
        filtered_df
        .groupby(
            filtered_df["order_date"]
            .dt.to_period("M")
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


    st.subheader(
        "📈 Historical Monthly Sales"
    )


    fig_history = px.line(
        monthly_sales,
        x="order_date",
        y="sales",
        title="Historical Sales Trend",
        labels={
            "order_date": "Month",
            "sales": "Sales"
        },
        markers=True
    )

    apply_chart_style(
        fig_history
    )

    st.plotly_chart(
        fig_history,
        use_container_width=True
    )


    # ========================================================
    # FORECAST
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔮 Next 12-Month Sales Forecast"
    )


    if len(monthly_sales) >= 2:

        monthly_sales["time_index"] = np.arange(
            len(monthly_sales)
        )

        X = monthly_sales[
            ["time_index"]
        ]

        y = monthly_sales[
            "sales"
        ]

        model = LinearRegression()

        model.fit(
            X,
            y
        )

        future_index = np.arange(
            len(monthly_sales),
            len(monthly_sales) + 12
        ).reshape(-1, 1)

        forecast_values = model.predict(
            future_index
        )

        forecast_values = np.maximum(
            forecast_values,
            0
        )

        last_month = (
            monthly_sales["order_date"]
            .max()
        )

        forecast_dates = pd.date_range(
            start=(
                last_month
                + pd.offsets.MonthBegin(1)
            ),
            periods=12,
            freq="MS"
        )

        forecast = pd.DataFrame({

            "Month": forecast_dates,

            "Predicted Sales":
                forecast_values

        })


        # ----------------------------------------------------
        # FORECAST KPIs
        # ----------------------------------------------------

        forecast_total = (
            forecast["Predicted Sales"]
            .sum()
        )

        forecast_average = (
            forecast["Predicted Sales"]
            .mean()
        )

        forecast_peak = (
            forecast["Predicted Sales"]
            .max()
        )

        forecast_peak_month = (
            forecast.loc[
                forecast["Predicted Sales"]
                .idxmax(),
                "Month"
            ]
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "🔮 Forecasted 12-Month Sales",
                currency(
                    forecast_total
                )
            )


        with col2:

            st.metric(
                "📊 Average Monthly Forecast",
                currency(
                    forecast_average
                )
            )


        with col3:

            st.metric(
                "🏆 Peak Forecast Month",
                forecast_peak_month.strftime(
                    "%b %Y"
                )
            )


        # ----------------------------------------------------
        # FORECAST CHART
        # ----------------------------------------------------

        fig_forecast = px.line(
            forecast,
            x="Month",
            y="Predicted Sales",
            title="🔮 Next 12-Month Sales Forecast",
            labels={
                "Month": "Month",
                "Predicted Sales":
                    "Predicted Sales"
            },
            markers=True
        )

        apply_chart_style(
            fig_forecast
        )

        st.plotly_chart(
            fig_forecast,
            use_container_width=True
        )


        # ----------------------------------------------------
        # FORECAST TABLE
        # ----------------------------------------------------

        st.subheader(
            "📋 Forecasted Sales"
        )

        display_forecast = (
            forecast.copy()
        )

        display_forecast["Month"] = (
            display_forecast["Month"]
            .dt.strftime("%b %Y")
        )

        display_forecast[
            "Predicted Sales"
        ] = (
            display_forecast[
                "Predicted Sales"
            ].round(2)
        )

        st.dataframe(
            display_forecast,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "⚠️ Not enough monthly data "
            "to generate a reliable forecast."
        )

        forecast = pd.DataFrame(
            columns=[
                "Month",
                "Predicted Sales"
            ]
        )


    # ========================================================
    # GEMINI AI
    # ========================================================

    st.markdown("---")

    st.header(
        "🤖 Gemini AI Business Assistant"
    )

    st.caption(
        "Use AI to understand your business data, "
        "find problems and make decisions."
    )


    # --------------------------------------------------------
    # PREPARE AI BUSINESS CONTEXT
    # --------------------------------------------------------

    category_context = (
        filtered_df
        .groupby("category")
        .agg(
            Sales=("sales", "sum"),
            Profit=("profit", "sum"),
            Quantity=("quantity", "sum")
        )
        .round(2)
        .sort_values(
            "Sales",
            ascending=False
        )
        .head(20)
        .to_string()
    )


    region_context = (
        filtered_df
        .groupby("region")
        .agg(
            Sales=("sales", "sum"),
            Profit=("profit", "sum")
        )
        .round(2)
        .sort_values(
            "Sales",
            ascending=False
        )
        .to_string()
    )


    segment_context = (
        filtered_df
        .groupby("segment")
        .agg(
            Sales=("sales", "sum"),
            Profit=("profit", "sum")
        )
        .round(2)
        .sort_values(
            "Sales",
            ascending=False
        )
        .to_string()
    )


    historical_context = (
        monthly_sales[
            ["order_date", "sales"]
        ]
        .tail(24)
        .to_string(index=False)
    )


    forecast_context = (
        forecast.to_string(
            index=False
        )
        if not forecast.empty
        else "Forecast unavailable."
    )


    business_context = f"""

RETAILPULSE AI BUSINESS CONTEXT

Selected Date Range:
{start_date.strftime("%d %b %Y")}
to
{end_date.strftime("%d %b %Y")}

TOTAL SALES:
{total_sales:.2f}

TOTAL PROFIT:
{total_profit:.2f}

TOTAL QUANTITY:
{total_quantity:.0f}

PROFIT MARGIN:
{profit_margin:.2f}%

TOTAL ORDERS:
{total_orders}

TOTAL CUSTOMERS:
{total_customers}

RECORDS:
{len(filtered_df)}

SELECTED CATEGORIES:
{", ".join(selected_categories)}

SELECTED REGIONS:
{", ".join(selected_regions)}

SELECTED CUSTOMER SEGMENTS:
{", ".join(selected_segments)}


CATEGORY PERFORMANCE:
{category_context}


REGION PERFORMANCE:
{region_context}


CUSTOMER SEGMENT PERFORMANCE:
{segment_context}


RECENT HISTORICAL MONTHLY SALES:
{historical_context}


NEXT 12-MONTH FORECAST:
{forecast_context}

"""


    # ========================================================
    # FIXED AI INSIGHTS
    # ========================================================

    with st.expander(
        "✨ Generate AI Business Insights",
        expanded=False
    ):

        st.write(
            "Generate a structured business analysis "
            "using the currently selected filters."
        )


        if st.button(
            "✨ Generate AI Insights",
            key="generate_ai_insights"
        ):

            try:

                from google import genai

                api_key = os.environ.get(
                    "GEMINI_API_KEY"
                )

                if not api_key:

                    st.error(
                        "❌ GEMINI_API_KEY is not configured."
                    )

                else:

                    client = genai.Client(
                        api_key=api_key
                    )


                    prompt = f"""

You are an expert retail business analyst
working for RetailPulse AI.

Analyze ONLY the business data provided below.

{business_context}

Provide the analysis in exactly these sections:

### 📈 TREND ANALYSIS

Analyze:

- Historical sales trends
- Profit trends
- Category performance
- Region performance
- Customer segment performance
- Next 12-month sales forecast

### 💡 BUSINESS INSIGHTS

Identify the most important findings
from the selected data.

### ❓ WHY IS THIS HAPPENING?

Explain possible reasons behind
the observed trends using ONLY
the available data.

Do not invent causes.

### 🎯 RECOMMENDATIONS

Give exactly 5 practical and
actionable recommendations.

RULES:

- Do not invent numbers.
- Use only the provided data.
- Respect the selected date range.
- Consider the selected filters.
- Clearly distinguish facts from assumptions.
- Do not claim correlation is causation.
- Keep explanations business-focused.
- Mention when something cannot be determined.
- Use simple and clear language.

"""


                    with st.spinner(
                        "🤖 Gemini is analyzing the business data..."
                    ):

                        response = (
                            client.models.generate_content(
                                model="gemini-3.5-flash-lite",
                                contents=prompt
                            )
                        )


                    st.success(
                        "✅ AI analysis generated successfully!"
                    )

                    st.markdown(
                        response.text
                    )


            except Exception as e:

                st.error(
                    "❌ Gemini AI could not generate the analysis."
                )

                st.exception(e)


    # ========================================================
    # CUSTOM USER PROMPT
    # ========================================================

    st.markdown("---")

    st.subheader(
        "💬 Ask Gemini Anything"
    )

    st.write(
        "Ask your own business question using the "
        "currently selected dashboard data."
    )


    user_prompt = st.text_area(
        "Your Question",
        placeholder=(
            "Example: Why is profit lower in the "
            "West region and what should the company do?"
        ),
        height=120,
        key="custom_gemini_prompt"
    )


    if st.button(
        "🤖 Ask Gemini",
        key="ask_custom_gemini"
    ):

        if not user_prompt.strip():

            st.warning(
                "⚠️ Please enter a question first."
            )

        else:

            try:

                from google import genai

                api_key = os.environ.get(
                    "GEMINI_API_KEY"
                )

                if not api_key:

                    st.error(
                        "❌ GEMINI_API_KEY is not configured."
                    )

                else:

                    client = genai.Client(
                        api_key=api_key
                    )


                    custom_prompt = f"""

You are the AI Business Assistant
inside RetailPulse AI.

The user has asked:

"{user_prompt}"

You must answer the user's question
using the retail business context below.

IMPORTANT:

- Use ONLY the provided data.
- Do not invent numbers.
- If the data is insufficient, say so.
- Respect the selected filters.
- Respect the selected date range.
- Use the forecast when relevant.
- Distinguish facts from assumptions.
- Do not claim causation without evidence.
- Give practical business-oriented answers.
- If useful, provide clear action steps.
- Keep the answer understandable for a business user.

RETAIL BUSINESS CONTEXT:

{business_context}

USER QUESTION:

{user_prompt}

"""


                    with st.spinner(
                        "🤖 Gemini is preparing your answer..."
                    ):

                        response = (
                            client.models.generate_content(
                                model="gemini-3.5-flash-lite",
                                contents=custom_prompt
                            )
                        )


                    st.success(
                        "✅ Gemini response generated!"
                    )

                    st.markdown(
                        response.text
                    )


            except Exception as e:

                st.error(
                    "❌ Gemini could not answer the question."
                )

                st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "© RetailPulse AI • AI-Powered Retail Analytics "
    "and Decision Support Platform"
)
