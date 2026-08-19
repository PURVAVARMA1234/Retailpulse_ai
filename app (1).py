import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. PAGE CONFIGURATION & SESSION STATE
# ============================================================
st.set_page_config(
    page_title="RetailPulse AI - Enterprise Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "sales_data" not in st.session_state:
    st.session_state["sales_data"] = pd.DataFrame()
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = "Retail_Dataset"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "ai_summary" not in st.session_state:
    st.session_state["ai_summary"] = "No AI analysis generated yet."

# Retrieve Gemini Key securely from Streamlit Secrets or Sidebar fallback
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", None)

# ============================================================
# 2. HELPER FUNCTIONS & CLEANING
# ============================================================
def clean_and_prep_dataframe(df):
    """Normalize column names and enforce datetime/numeric formats."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    mapping = {
        "order_id": "order_id",
        "customer_id": "customer_id",
        "order_date": "order_date",
        "date": "order_date",
        "sales": "sales",
        "revenue": "sales",
        "profit": "profit",
        "region": "region",
        "category": "category",
        "sub_category": "sub_category",
        "segment": "segment"
    }

    for col in list(df.columns):
        if col in mapping:
            df.rename(columns={col: mapping[col]}, inplace=True)

    defaults = {
        "order_id": lambda: [f"ORD-{i}" for i in range(len(df))],
        "customer_id": lambda: [f"CUST-{i%500}" for i in range(len(df))],
        "sales": 0.0,
        "profit": 0.0,
        "region": "General",
        "category": "Uncategorized",
        "sub_category": "General Item",
        "segment": "Standard"
    }

    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val() if callable(default_val) else default_val

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0.0)
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0.0)
    
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    else:
        df["order_date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")

    # Fallback for null dates post-conversion
    if df["order_date"].isnull().any():
        df["order_date"] = df["order_date"].fillna(pd.Timestamp("2024-01-01"))

    df["profit_margin"] = np.where(df["sales"] > 0, (df["profit"] / df["sales"]) * 100, 0.0)
    return df

def generate_editable_html_report(df, ai_summary, dataset_name):
    """Generate printable, fully editable HTML report with inline charts."""
    total_sales = df['sales'].sum()
    total_profit = df['profit'].sum()
    margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    total_orders = len(df)

    region_summary = df.groupby("region")["sales"].sum().reset_index()
    category_summary = df.groupby("category")[["sales", "profit"]].sum().reset_index()

    reg_rows = "".join([f"<tr><td>{r['region']}</td><td>${r['sales']:,.2f}</td></tr>" for _, r in region_summary.iterrows()])
    cat_rows = "".join([f"<tr><td>{r['category']}</td><td>${r['sales']:,.2f}</td><td>${r['profit']:,.2f}</td></tr>" for _, r in category_summary.iterrows()])

    # Generate CSS-based visual bar chart for report
    max_reg_sales = region_summary["sales"].max() if not region_summary.empty else 1
    chart_bars = ""
    for _, r in region_summary.iterrows():
        width_pct = (r['sales'] / max_reg_sales) * 100
        chart_bars += f"""
        <div style="margin-bottom: 8px;">
            <div style="font-size: 13px; font-weight: bold;">{r['region']} (${r['sales']:,.2f})</div>
            <div style="background-color: #2563EB; height: 18px; width: {width_pct:.1f}%; border-radius: 4px;"></div>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{dataset_name} - Executive Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; background-color: #FAFAFA; }}
            .report-card {{ background: #FFFFFF; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            h1 {{ color: #0F172A; border-bottom: 3px solid #2563EB; padding-bottom: 8px; }}
            .editable-badge {{ background-color: #FEF08A; color: #854D0E; font-size: 12px; padding: 4px 8px; border-radius: 4px; float: right; }}
            .kpi-row {{ display: flex; gap: 15px; margin: 20px 0; }}
            .kpi-card {{ background: #F8FAFC; border: 1px solid #E2E8F0; padding: 15px; border-radius: 8px; flex: 1; text-align: center; }}
            .kpi-val {{ font-size: 20px; font-weight: bold; color: #1E40AF; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #CBD5E1; padding: 10px; text-align: left; }}
            th {{ background: #F1F5F9; }}
            .ai-box {{ background: #EFF6FF; border-left: 4px solid #2563EB; padding: 15px; margin-top: 20px; border-radius: 4px; }}
            .chart-box {{ background: #F8FAFC; border: 1px solid #E2E8F0; padding: 15px; border-radius: 8px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="report-card" contenteditable="true">
            <span class="editable-badge">✏️ Fully Editable Report (Click text to edit)</span>
            <h1>🛍️ Executive Performance Report: {dataset_name}</h1>
            <p><strong>Generated Date:</strong> {pd.Timestamp.now().strftime('%B %d, %Y')}</p>
            
            <div class="kpi-row">
                <div class="kpi-card"><div>Total Revenue</div><div class="kpi-val">${total_sales:,.2f}</div></div>
                <div class="kpi-card"><div>Total Profit</div><div class="kpi-val">${total_profit:,.2f}</div></div>
                <div class="kpi-card"><div>Profit Margin</div><div class="kpi-val">{margin:.2f}%</div></div>
                <div class="kpi-card"><div>Total Orders</div><div class="kpi-val">{total_orders:,}</div></div>
            </div>

            <h2>📊 Regional Visual Breakdown</h2>
            <div class="chart-box">
                {chart_bars}
            </div>

            <h2>📍 Regional Data Table</h2>
            <table><thead><tr><th>Region</th><th>Sales</th></tr></thead><tbody>{reg_rows}</tbody></table>

            <h2>🏷️ Category Performance</h2>
            <table><thead><tr><th>Category</th><th>Sales</th><th>Profit</th></tr></thead><tbody>{cat_rows}</tbody></table>

            <h2>🤖 AI Strategic Recommendations</h2>
            <div class="ai-box">{ai_summary.replace('\n', '<br>')}</div>
        </div>
    </body>
    </html>
    """

# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🛍️ RetailPulse AI")

# Fallback API key input in sidebar if not in secrets
if not GEMINI_KEY:
    user_key = st.sidebar.text_input("Gemini API Key (Optional):", type="password")
    if user_key:
        GEMINI_KEY = user_key

menu = st.sidebar.radio(
    "Navigation Engine",
    [
        "📂 Data Ingestion Hub",
        "📊 Executive Overview",
        "💬 Gemini Conversational Assistant",
        "📥 Report & Export Center"
    ]
)

df = st.session_state["sales_data"]

# ============================================================
# 4. MODULE 1: DATA INGESTION
# ============================================================
if menu == "📂 Data Ingestion Hub":
    st.header("📂 Data Ingestion Engine")
    uploaded_file = st.file_uploader("Upload Raw Sales File (CSV or Excel)", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        if st.button("🚀 Ingest & Clean Dataset"):
            with st.spinner("Processing & normalizing data..."):
                try:
                    clean_name = str(uploaded_file.name).split(".")[0].replace(" ", "_")
                    st.session_state["dataset_name"] = clean_name

                    raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                    st.session_state["sales_data"] = clean_and_prep_dataframe(raw_df)
                    st.success(f"✅ Loaded dataset **{clean_name}** with {len(st.session_state['sales_data']):,} records.")
                    st.dataframe(st.session_state["sales_data"].head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")

    elif not df.empty:
        st.info(f"Active Session Dataset: **{st.session_state['dataset_name']}** ({len(df):,} records)")

# ============================================================
# 5. MODULE 2: EXECUTIVE OVERVIEW (FIXED RESAMPLE ERROR)
# ============================================================
elif menu == "📊 Executive Overview":
    st.header("📊 Executive Performance Dashboard")
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"${df['sales'].sum():,.2f}")
        col2.metric("Total Profit", f"${df['profit'].sum():,.2f}")
        col3.metric("Profit Margin", f"{(df['profit'].sum()/df['sales'].sum()*100):.2f}%")
        col4.metric("Avg Order Value", f"${df['sales'].mean():,.2f}")

        st.markdown("---")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Regional Revenue Performance")
            reg_df = df.groupby("region")["sales"].sum().reset_index()
            fig_reg = px.bar(reg_df, x="region", y="sales", color="region", template="plotly_white", text_auto=".2s")
            st.plotly_chart(fig_reg, use_container_width=True)

        with col_right:
            st.subheader("Sales Trend Over Time")
            # FIXED: Safe Datetime resampling with Pandas 2.0+ compatibility ('ME' instead of deprecated 'M')
            try:
                time_df = df.dropna(subset=["order_date"]).set_index("order_date").resample("ME")["sales"].sum().reset_index()
                fig_line = px.line(time_df, x="order_date", y="sales", markers=True, template="plotly_white")
                st.plotly_chart(fig_line, use_container_width=True)
            except Exception as e:
                # Fallback to simple date grouping if resampling fails
                time_df = df.groupby(df["order_date"].dt.date)["sales"].sum().reset_index()
                fig_line = px.line(time_df, x="order_date", y="sales", template="plotly_white")
                st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("⚠️ Please upload a dataset in Tab 1 first.")

# ============================================================
# 6. MODULE 3: CONVERSATIONAL GEMINI CHAT
# ============================================================
elif menu == "💬 Gemini Conversational Assistant":
    st.header("💬 Gemini Executive AI Advisor")
    
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a question about your business performance..."):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if not GEMINI_KEY:
            reply = "⚠️ API Key is missing. Please configure GEMINI_API_KEY in Streamlit Secrets or enter it in the sidebar."
        elif df.empty:
            reply = "⚠️ Please upload a dataset in Tab 1 first so I can analyze it."
        else:
            with st.spinner("Gemini is thinking..."):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=GEMINI_KEY)
                    
                    context = f"Sales: ${df['sales'].sum():,.2f}, Profit: ${df['profit'].sum():,.2f}, Categories: {df['category'].unique().tolist()}"
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(f"Context: {context}\nQuestion: {prompt}")
                    reply = res.text
                    st.session_state["ai_summary"] = reply
                except Exception as e:
                    reply = f"AI Service Error: {str(e)}"

        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

# ============================================================
# 7. MODULE 4: REPORT & EXPORT CENTER
# ============================================================
elif menu == "📥 Report & Export Center":
    st.header("📥 Report & Data Export Hub")
    if not df.empty:
        dataset_name = st.session_state.get("dataset_name", "RetailPulse_Report")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📄 Export Processed CSV")
            st.download_button(
                "Download Clean CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{dataset_name}_clean.csv",
                mime="text/csv"
            )

        with c2:
            st.subheader("📊 Download Editable Executive Report")
            html_rep = generate_editable_html_report(df, st.session_state["ai_summary"], dataset_name)
            
            st.download_button(
                "📄 Download Executive Report (.html)",
                data=html_rep,
                file_name=f"{dataset_name}_Executive_Report.html",
                mime="text/html"
            )
            st.caption("✨ *Report Features: Embedded visual charts, dynamic filename matching your dataset, and fully editable text/tables directly in browser.*")
    else:
        st.warning("⚠️ No dataset available. Please upload a file in Tab 1.")
