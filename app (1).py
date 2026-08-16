import io
import re
import urllib.parse
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine

# ============================================================
# 1. PAGE CONFIGURATION & DATABASE ENGINE SETUP
# ============================================================
st.set_page_config(
    page_title="RetailPulse AI Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PostgreSQL Connection Engine setup
safe_password = urllib.parse.quote_plus("purva@001")
DATABASE_URL = f"postgresql+psycopg2://postgres:{safe_password}@localhost:5432/retailpulse_api"
engine = create_engine(DATABASE_URL)

@st.cache_data(ttl=60)
def load_data():
    """Fetch ingested sales dataset directly from PostgreSQL database."""
    try:
        return pd.read_sql("SELECT * FROM uploaded_company_sales", con=engine)
    except Exception:
        return pd.DataFrame()


# ============================================================
# 2. NAVIGATION MENU & SIDEBAR
# ============================================================
st.title("🛍️ RetailPulse AI: Enterprise Dashboard & Ingestion Hub")

menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "📂 Upload Company Data",
        "📊 Executive Overview",
        "📈 Sales & Customer Analytics",
        "🔮 Forecast & Gemini AI",
        "📥 Export & Reports"
    ]
)


# ============================================================
# 3. TAB 1: UPLOAD COMPANY DATA (FASTAPI CONNECTOR)
# ============================================================
if menu == "📂 Upload Company Data":
    st.header("📂 Dynamic CSV / XLSX Data Ingestion Engine")
    st.write("Upload raw sales data to process via FastAPI and automatically store into PostgreSQL `retailpulse_api`.")

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        st.info(f"File Selected: **{uploaded_file.name}**")
        if st.button("🚀 Process & Ingest to Database"):
            with st.spinner("Processing file via FastAPI backend..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post("http://127.0.0.1:8000/upload", files=files)

                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"✅ Data Ingested Successfully! Total Rows Imported: {res_data['total_rows_imported']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Mapped Target Schema:**")
                            st.json(res_data["mapped_columns"])
                        with col2:
                            st.write("**Unmapped Preserved Columns:**")
                            st.write(res_data["unmapped_columns"])
                    else:
                        st.error(f"❌ Ingestion Failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ API Connection Error: {str(e)}")


# ============================================================
# 4. TAB 2: EXECUTIVE OVERVIEW
# ============================================================
elif menu == "📊 Executive Overview":
    st.header("📊 Executive Overview")
    df = load_data()

    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"${df['sales'].sum():,.2f}")
        col2.metric("Total Profit", f"${df['profit'].sum():,.2f}")
        col3.metric("Total Orders", f"{len(df):,}")
        col4.metric("Avg Order Value", f"${df['sales'].mean():,.2f}")

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("Regional Revenue Breakdown")
            st.bar_chart(data=df.groupby("region")["sales"].sum().reset_index(), x="region", y="sales")
        with chart_col2:
            st.subheader("Category Revenue Breakdown")
            st.bar_chart(data=df.groupby("category")["sales"].sum().reset_index(), x="category", y="sales")
    else:
        st.warning("⚠️ Database is empty. Please upload data first!")


# ============================================================
# 5. TAB 3: SALES & CUSTOMER ANALYTICS
# ============================================================
elif menu == "📈 Sales & Customer Analytics":
    st.header("📈 Sales & Customer Analytics")
    df = load_data()

    if not df.empty:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_region = st.multiselect(
                "Select Region(s):",
                options=df["region"].unique().tolist(),
                default=df["region"].unique().tolist()
            )
        with col_f2:
            selected_category = st.multiselect(
                "Select Category(ies):",
                options=df["category"].unique().tolist(),
                default=df["category"].unique().tolist()
            )

        filtered_df = df[
            (df["region"].isin(selected_region)) & 
            (df["category"].isin(selected_category))
        ]

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.write("**Sales by Category**")
            st.bar_chart(data=filtered_df.groupby("category")["sales"].sum().reset_index(), x="category", y="sales")
        with chart_col2:
            st.write("**Profit by Segment**")
            st.bar_chart(data=filtered_df.groupby("segment")["profit"].sum().reset_index(), x="segment", y="profit")

        st.subheader("📋 Dataset Preview")
        st.dataframe(filtered_df.head(100), use_container_width=True)
    else:
        st.warning("⚠️ No data available. Ingest dataset first.")


# ============================================================
# 6. TAB 4: FORECAST & GEMINI AI ASSISTANT
# ============================================================
elif menu == "🔮 Forecast & Gemini AI":
    st.header("🔮 Sales Forecast & Gemini Executive Assistant")
    df = load_data()

    if not df.empty:
        # --- GEMINI AI INTEGRATION ---
        st.subheader("🤖 Ask Gemini Executive Consultant")
        api_key = st.text_input("Enter Gemini API Key:", type="password")
        user_prompt = st.text_area(
            "Ask a strategic question:",
            placeholder="e.g., Which category is generating high sales but low profit margin?"
        )

        if st.button("✨ Generate AI Insights"):
            if api_key and user_prompt:
                with st.spinner("Analyzing dataset with Gemini AI..."):
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=api_key)

                        data_context = (
                            f"Total Records: {len(df)}\n"
                            f"Total Sales: ${df['sales'].sum():,.2f}\n"
                            f"Total Profit: ${df['profit'].sum():,.2f}\n\n"
                            f"Category Breakdown:\n" + 
                            df.groupby("category")[["sales", "profit"]].sum().to_string()
                        )

                        prompt = f"""
                        You are an executive retail analyst. Based on this data context:
                        {data_context}

                        Question: {user_prompt}
                        Provide structured business recommendations.
                        """

                        model = genai.GenerativeModel("gemini-1.5-flash")
                        res = model.generate_content(prompt)

                        st.success("Analysis Complete!")
                        st.markdown("### Executive Insights:")
                        st.write(res.text)
                    except Exception as e:
                        st.error(f"AI Error: {str(e)}")
            else:
                st.warning("Please provide both API Key and your Question.")

        st.markdown("---")

        # --- PROPHET FORECASTING ENGINE ---
        st.subheader("📈 Time-Series Sales Forecasting (Prophet ML)")
        horizon = st.slider("Select Forecast Horizon (Days):", 7, 90, 30)

        if st.button("🔮 Run Forecast Model"):
            with st.spinner("Training Prophet time-series model..."):
                try:
                    from prophet import Prophet
                    if "order_date" in df.columns:
                        df['ds'] = pd.to_datetime(df['order_date'])
                        daily = df.groupby('ds')['sales'].sum().reset_index()
                        daily.columns = ['ds', 'y']

                        m = Prophet(daily_seasonality=True)
                        m.fit(daily)

                        future = m.make_future_dataframe(periods=horizon)
                        fcst = m.predict(future)

                        st.line_chart(fcst.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].tail(horizon + 30))
                        st.success("Forecast Model Generated Successfully!")
                    else:
                        st.error("Column 'order_date' missing in PostgreSQL schema.")
                except Exception as e:
                    st.error(f"Forecast Engine Error: {str(e)}")
    else:
        st.warning("⚠️ No data found in database. Ingest dataset first.")


# ============================================================
# 7. TAB 5: DATA EXPORT ENGINE
# ============================================================
elif menu == "📥 Export & Reports":
    st.header("📥 Data Export Engine")
    df = load_data()

    if not df.empty:
        st.write("Download processed database records as CSV:")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Clean CSV Dataset",
            data=csv_data,
            file_name="retailpulse_processed_sales.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Database is empty. Ingest data to export.")
