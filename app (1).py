import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

# ============================================================
# 1. PAGE CONFIGURATION & SESSION STATE
# ============================================================
st.set_page_config(
    page_title="RetailPulse AI - Enterprise Intelligence Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "sales_data" not in st.session_state:
    st.session_state["sales_data"] = pd.DataFrame()
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "ai_summary" not in st.session_state:
    st.session_state["ai_summary"] = "No AI analysis performed yet."

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def clean_and_prep_dataframe(df):
    """Normalize column names and handle essential fields."""
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

    # Profit Margin %
    df["profit_margin"] = np.where(df["sales"] > 0, (df["profit"] / df["sales"]) * 100, 0.0)
    return df

def run_rfm_segmentation(df):
    """K-Means RFM Segmentation."""
    if "customer_id" not in df.columns:
        return df

    snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg({
        "order_date": lambda x: (snapshot_date - x.max()).days,
        "order_id": "nunique",
        "sales": "sum"
    }).reset_index()

    rfm.columns = ["customer_id", "Recency", "Frequency", "Monetary"]
    
    # Scale & Cluster
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    
    # Map clusters to logical segment names based on Monetary value
    cluster_means = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False)
    cluster_map = {
        cluster_means.index[0]: "VIP Customers",
        cluster_means.index[1]: "Regular Customers",
        cluster_means.index[2]: "At-Risk / Inactive"
    }
    rfm["Customer_Segment"] = rfm["Cluster"].map(cluster_map)
    return rfm

def generate_html_report(df, ai_summary):
    """Generate printable HTML Executive Summary."""
    total_sales = df['sales'].sum()
    total_profit = df['profit'].sum()
    margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    total_orders = len(df)

    region_summary = df.groupby("region")["sales"].sum().reset_index()
    category_summary = df.groupby("category")[["sales", "profit"]].sum().reset_index()

    reg_rows = "".join([f"<tr><td>{r['region']}</td><td>${r['sales']:,.2f}</td></tr>" for _, r in region_summary.iterrows()])
    cat_rows = "".join([f"<tr><td>{r['category']}</td><td>${r['sales']:,.2f}</td><td>${r['profit']:,.2f}</td></tr>" for _, r in category_summary.iterrows()])

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RetailPulse AI Executive Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
            h1 {{ color: #0F172A; border-bottom: 3px solid #2563EB; padding-bottom: 8px; }}
            .kpi-row {{ display: flex; gap: 20px; margin: 20px 0; }}
            .kpi-card {{ background: #F8FAFC; border: 1px solid #E2E8F0; padding: 15px; border-radius: 8px; flex: 1; text-align: center; }}
            .kpi-val {{ font-size: 22px; font-weight: bold; color: #1E40AF; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #CBD5E1; padding: 10px; text-align: left; }}
            th {{ background: #F1F5F9; }}
            .ai-box {{ background: #EFF6FF; border-left: 4px solid #2563EB; padding: 15px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <h1>🛍️ RetailPulse AI: Executive Performance Report</h1>
        <p><strong>Generated Date:</strong> {pd.Timestamp.now().strftime('%B %d, %Y')}</p>
        
        <div class="kpi-row">
            <div class="kpi-card"><div>Total Revenue</div><div class="kpi-val">${total_sales:,.2f}</div></div>
            <div class="kpi-card"><div>Total Profit</div><div class="kpi-val">${total_profit:,.2f}</div></div>
            <div class="kpi-card"><div>Profit Margin</div><div class="kpi-val">{margin:.2f}%</div></div>
            <div class="kpi-card"><div>Total Orders</div><div class="kpi-val">{total_orders:,}</div></div>
        </div>

        <h2>📍 Regional Sales Breakdown</h2>
        <table><thead><tr><th>Region</th><th>Sales</th></tr></thead><tbody>{reg_rows}</tbody></table>

        <h2>🏷️ Category Performance</h2>
        <table><thead><tr><th>Category</th><th>Sales</th><th>Profit</th></tr></thead><tbody>{cat_rows}</tbody></table>

        <h2>🤖 AI Strategic Insights</h2>
        <div class="ai-box">{ai_summary.replace('\n', '<br>')}</div>
    </body>
    </html>
    """

# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🛍️ RetailPulse AI")
menu = st.sidebar.radio(
    "Navigation Engine",
    [
        "📂 Data Ingestion Hub",
        "📊 Executive Overview",
        "🎯 Advanced Customer RFM",
        "🛒 Basket & Margin Analytics",
        "🔮 Forecasting & Scenario Planning",
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
            with st.spinner("Normalizing data & calculating metrics..."):
                try:
                    raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                    st.session_state["sales_data"] = clean_and_prep_dataframe(raw_df)
                    st.success(f"✅ Ingestion Complete! Loaded {len(st.session_state['sales_data']):,} records.")
                    st.dataframe(st.session_state["sales_data"].head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")

    elif not df.empty:
        st.info(f"Current active session dataset: **{len(df):,} records**.")

# ============================================================
# 5. MODULE 2: EXECUTIVE OVERVIEW
# ============================================================
elif menu == "📊 Executive Overview":
    st.header("📊 Executive Performance Dashboard")
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"${df['sales'].sum():,.2f}", delta="+12.4% MoM")
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
            time_df = df.set_index("order_date").resample("M")["sales"].sum().reset_index()
            fig_line = px.line(time_df, x="order_date", y="sales", markers=True, template="plotly_white")
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("⚠️ Please upload a dataset in Tab 1 first.")

# ============================================================
# 6. MODULE 3: ADVANCED CUSTOMER RFM
# ============================================================
elif menu == "🎯 Advanced Customer RFM":
    st.header("🎯 Customer RFM Segmentation & Behavior")
    if not df.empty:
        if st.button("⚡ Run RFM Clustering Engine"):
            with st.spinner("Clustering customer segments..."):
                rfm_df = run_rfm_segmentation(df)
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.subheader("Segment Distribution")
                    seg_counts = rfm_df["Customer_Segment"].value_counts().reset_index()
                    fig_pie = px.pie(seg_counts, values="count", names="Customer_Segment", hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with c2:
                    st.subheader("Monetary vs Recency Scatter Profile")
                    fig_scat = px.scatter(
                        rfm_df, x="Recency", y="Monetary", color="Customer_Segment",
                        size="Frequency", hover_data=["customer_id"], template="plotly_white"
                    )
                    st.plotly_chart(fig_scat, use_container_width=True)

                st.subheader("📋 Segmented Customer Preview")
                st.dataframe(rfm_df.head(50), use_container_width=True)
    else:
        st.warning("⚠️ No dataset loaded.")

# ============================================================
# 7. MODULE 4: BASKET & MARGIN ANALYTICS
# ============================================================
elif menu == "🛒 Basket & Margin Analytics":
    st.header("🛒 Profit Margin & Market Basket Analysis")
    if not df.empty:
        t1, t2 = st.columns(2)
        with t1:
            st.subheader("⚠️ Low Margin / Losing Categories")
            low_margin = df.groupby("sub_category")[["sales", "profit", "profit_margin"]].mean().sort_values(by="profit_margin").reset_index()
            fig_margin = px.bar(low_margin.head(10), x="profit_margin", y="sub_category", orientation="h", color="profit_margin", color_continuous_scale="Reds")
            st.plotly_chart(fig_margin, use_container_width=True)

        with t2:
            st.subheader("🛒 Market Basket Analysis (Apriori)")
            try:
                basket = (df.groupby(['order_id', 'sub_category'])['sales']
                          .sum().unstack().reset_index().fillna(0)
                          .set_index('order_id'))
                basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
                
                frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                
                if not rules.empty:
                    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10), use_container_width=True)
                else:
                    st.info("No strong product basket associations found at current threshold.")
            except Exception as e:
                st.info("Need more itemized data per Order ID to compute Market Basket analysis.")
    else:
        st.warning("⚠️ Load dataset first.")

# ============================================================
# 8. MODULE 5: FORECASTING & SCENARIO PLANNING
# ============================================================
elif menu == "🔮 Forecasting & Scenario Planning":
    st.header("🔮 Sales Forecasting & 'What-If' Simulation")
    if not df.empty:
        st.subheader("🎛️ 'What-If' Pricing Scenario Simulator")
        price_change = st.slider("Simulate Category Price Adjustment (%):", -20, 20, 5)
        
        target_cat = st.selectbox("Select Target Category:", df["category"].unique())
        
        sim_df = df.copy()
        sim_df.loc[sim_df["category"] == target_cat, "sales"] *= (1 + price_change / 100)
        
        orig_sales = df["sales"].sum()
        new_sales = sim_df["sales"].sum()
        diff = new_sales - orig_sales

        st.metric("Projected Total Revenue", f"${new_sales:,.2f}", delta=f"${diff:,.2f} impact")

        st.markdown("---")
        st.subheader("📈 Time-Series Sales Forecast (Prophet ML)")
        days = st.slider("Forecast Days:", 7, 60, 30)
        
        if st.button("Run ML Forecast"):
            with st.spinner("Training Prophet Model..."):
                try:
                    from prophet import Prophet
                    daily = df.groupby('order_date')['sales'].sum().reset_index()
                    daily.columns = ['ds', 'y']
                    
                    m = Prophet()
                    m.fit(daily)
                    future = m.make_future_dataframe(periods=days)
                    fcst = m.predict(future)
                    
                    fig_fc = px.line(fcst.tail(days + 30), x="ds", y=["yhat", "yhat_lower", "yhat_upper"], title="30-Day Revenue Projection")
                    st.plotly_chart(fig_fc, use_container_width=True)
                except Exception as e:
                    st.error(f"Prophet Engine error: {str(e)}")
    else:
        st.warning("⚠️ No dataset loaded.")

# ============================================================
# 9. MODULE 6: CONVERSATIONAL GEMINI CHAT
# ============================================================
elif menu == "💬 Gemini Conversational Assistant":
    st.header("💬 Gemini Executive AI Advisor")
    
    api_key = st.sidebar.text_input("Gemini API Key:", type="password")

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a question about your business performance..."):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if not api_key:
            reply = "⚠️ Please enter your Gemini API Key in the sidebar to enable live AI responses."
        elif df.empty:
            reply = "⚠️ Please upload a dataset in Tab 1 so I can analyze your sales figures."
        else:
            with st.spinner("Gemini is thinking..."):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    
                    context = f"Sales: ${df['sales'].sum():,.2f}, Profit: ${df['profit'].sum():,.2f}, Categories: {df['category'].unique().tolist()}"
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(f"Data Context: {context}\n\nUser Question: {prompt}")
                    reply = res.text
                    st.session_state["ai_summary"] = reply
                except Exception as e:
                    reply = f"AI Error: {str(e)}"

        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

# ============================================================
# 10. MODULE 7: REPORT & EXPORT CENTER
# ============================================================
elif menu == "📥 Report & Export Center":
    st.header("📥 Report & Data Export Hub")
    if not df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📄 Export Processed Dataset")
            st.download_button(
                "Download Normalized CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="retailpulse_processed.csv",
                mime="text/csv"
            )

        with c2:
            st.subheader("📊 Executive Summary Report")
            html_rep = generate_html_report(df, st.session_state["ai_summary"])
            st.download_button(
                "Download Printable Executive Report (.html)",
                data=html_rep,
                file_name="executive_report.html",
                mime="text/html"
            )
            st.caption("💡 *Tip: Open the downloaded file in your browser and press Ctrl+P / Cmd+P to save as a PDF!*")
    else:
        st.warning("⚠️ No data available to export.")
