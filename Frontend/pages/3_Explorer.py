"""Customer Explorer Page."""

import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Customer Explorer - AutoML Advisor",
    page_icon="👥",
    layout="wide"
)

API_URL = "http://localhost:8000/api"

st.title("👥 Customer Explorer")
st.markdown("Explore individual customer predictions and risk factors.")

# Check for session
if "session_id" not in st.session_state:
    st.warning("⚠️ No analysis session found. Please upload data first.")
    if st.button("📤 Go to Upload"):
        st.switch_page("pages/1_Upload.py")
    st.stop()

session_id = st.session_state.session_id

# Filters
st.markdown("### 🔍 Filters")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    risk_filter = st.selectbox(
        "Risk Tier",
        options=["All", "High", "Medium", "Low"],
        index=0
    )

with filter_col2:
    limit = st.select_slider(
        "Results per page",
        options=[25, 50, 100, 200],
        value=50
    )

with filter_col3:
    page = st.number_input("Page", min_value=1, value=1)

# Fetch predictions
@st.cache_data(ttl=30)
def fetch_predictions(session_id, risk_tier, limit, offset):
    try:
        params = {"limit": limit, "offset": offset}
        if risk_tier and risk_tier != "All":
            params["risk_tier"] = risk_tier.lower()
        
        response = requests.get(f"{API_URL}/predictions/{session_id}", params=params)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

offset = (page - 1) * limit
risk_tier_param = risk_filter if risk_filter != "All" else None

predictions = fetch_predictions(session_id, risk_tier_param, limit, offset)

if not predictions:
    st.error("Could not load predictions. Please try again.")
    if st.button("🔄 Retry"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

# Display results
total = predictions.get("total", 0)
customers = predictions.get("customers", [])

st.markdown(f"### 📋 Customer List ({total:,} total)")

if customers:
    # Create DataFrame
    df = pd.DataFrame(customers)
    
    # Format columns
    df["churn_probability"] = (df["churn_probability"] * 100).round(1)
    
    # Add risk indicator
    def risk_indicator(tier):
        colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        return colors.get(tier, "⚪")
    
    df["risk_indicator"] = df["risk_tier"].apply(risk_indicator)
    
    # Reorder columns
    display_cols = ["risk_indicator", "customer_id", "churn_probability", "risk_tier"]
    if "clv" in df.columns:
        display_cols.append("clv")
    if "days_since_purchase" in df.columns:
        display_cols.append("days_since_purchase")
    
    display_df = df[display_cols].copy()
    display_df.columns = ["Risk", "Customer ID", "Churn Prob (%)", "Risk Tier"] + \
                        (["CLV ($)"] if "clv" in df.columns else []) + \
                        (["Days Since Purchase"] if "days_since_purchase" in df.columns else [])
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Churn Prob (%)": st.column_config.ProgressColumn(
                "Churn Prob (%)",
                min_value=0,
                max_value=100,
                format="%d%%"
            )
        }
    )
    
    # Pagination info
    total_pages = (total + limit - 1) // limit
    st.caption(f"Page {page} of {total_pages} • Showing {len(customers)} of {total:,} customers")
    
    # Customer detail view
    st.markdown("---")
    st.markdown("### 🔎 Customer Detail View")
    
    selected_customer = st.selectbox(
        "Select a customer to view details",
        options=[c["customer_id"] for c in customers],
        format_func=lambda x: f"{x} ({next((c['risk_tier'] for c in customers if c['customer_id'] == x), '')} risk)"
    )
    
    if selected_customer:
        customer_data = next((c for c in customers if c["customer_id"] == selected_customer), None)
        
        if customer_data:
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("#### Customer Profile")
                
                risk_tier = customer_data.get("risk_tier", "unknown")
                risk_colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#2ecc71"}
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {risk_colors.get(risk_tier, '#95a5a6')}22, {risk_colors.get(risk_tier, '#95a5a6')}11); 
                            border-left: 4px solid {risk_colors.get(risk_tier, '#95a5a6')};
                            padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                    <h3 style="margin: 0; color: {risk_colors.get(risk_tier, '#95a5a6')};">
                        {risk_tier.upper()} RISK
                    </h3>
                    <p style="font-size: 24px; margin: 5px 0;">
                        {customer_data.get('churn_probability', 0) * 100:.1f}% churn probability
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric(
                    "Customer ID",
                    customer_data.get("customer_id", "N/A")
                )
                
                if "clv" in customer_data:
                    st.metric(
                        "Customer Lifetime Value",
                        f"${customer_data.get('clv', 0):,.0f}"
                    )
                
                if "days_since_purchase" in customer_data:
                    days = customer_data.get("days_since_purchase", 0)
                    st.metric(
                        "Days Since Last Purchase",
                        f"{days} days",
                        delta=f"{'⚠️ Long absence' if days > 60 else '✅ Recent activity'}"
                    )
            
            with detail_col2:
                st.markdown("#### Recommended Actions")
                
                prob = customer_data.get("churn_probability", 0)
                
                if prob >= 0.7:
                    st.error("🚨 **Immediate Action Required**")
                    st.markdown("""
                    - Send personalized discount offer (20% off)
                    - Schedule manager outreach call
                    - Offer exclusive loyalty perks
                    """)
                elif prob >= 0.4:
                    st.warning("⚠️ **Prevention Recommended**")
                    st.markdown("""
                    - Send engagement email with recommendations
                    - Offer small incentive (10% off next purchase)
                    - Add to re-engagement campaign
                    """)
                else:
                    st.success("✅ **Customer Healthy**")
                    st.markdown("""
                    - Continue regular engagement
                    - Include in loyalty program communications
                    - Consider for referral campaigns
                    """)

else:
    st.info("No customers found matching the filters.")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Quick Stats")
    
    if predictions:
        st.metric("Showing", f"{len(customers)} customers")
        st.metric("Total Matching", f"{total:,}")
    
    st.divider()
    
    st.markdown("### 🔗 Quick Links")
    if st.button("📊 View Dashboard"):
        st.switch_page("pages/2_Dashboard.py")
    if st.button("🎯 What-If Simulator"):
        st.switch_page("pages/4_WhatIf.py")
