"""Dashboard Page - Business Insights & Actions."""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dashboard - AutoML Advisor",
    page_icon="📊",
    layout="wide"
)

API_URL = "http://localhost:8000/api"

# Check for session
if "session_id" not in st.session_state:
    st.warning("⚠️ No analysis session found. Please upload data first.")
    if st.button("📤 Go to Upload"):
        st.switch_page("pages/1_Upload.py")
    st.stop()

session_id = st.session_state.session_id

# Fetch results
@st.cache_data(ttl=60)
def fetch_results(session_id):
    try:
        response = requests.get(f"{API_URL}/results/{session_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

results = fetch_results(session_id)

if not results:
    st.error("Could not load results. Please try again.")
    if st.button("🔄 Retry"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

# Extract data
summary = results.get("summary", {})
feature_importance = results.get("feature_importance", [])
segments = results.get("segments", {})
drivers = results.get("drivers", [])
recommendations = results.get("recommendations", [])
roi = results.get("roi", {})

# ====================
# HEADER - Key Metrics
# ====================
st.title("📊 Customer Retention Dashboard")
st.markdown("*Your actionable insights to reduce customer churn and protect revenue*")

st.markdown("---")

# Big metric cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    high_risk = summary.get('high_risk_count', 0)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #e74c3c, #c0392b); padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">{high_risk}</h1>
        <p style="color: #ffcccc; margin: 0;">🚨 Customers at Risk</p>
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    revenue_at_risk = summary.get('revenue_at_risk', 0)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #e67e22, #d35400); padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">${revenue_at_risk:,.0f}</h1>
        <p style="color: #ffe0cc; margin: 0;">💰 Revenue at Risk</p>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    churn_rate = summary.get("churn_rate", 0) * 100
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #9b59b6, #8e44ad); padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">{churn_rate:.0f}%</h1>
        <p style="color: #e8d5f0; margin: 0;">📉 Predicted Churn Rate</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_customers = summary.get('total_customers', 0)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2ecc71, #27ae60); padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">{total_customers}</h1>
        <p style="color: #d5f5e3; margin: 0;">👥 Total Customers Analyzed</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ====================
# PRIORITY ACTIONS
# ====================
st.markdown("## 🎯 Priority Actions")
st.markdown("*Take these actions today to protect your revenue*")

action_col1, action_col2 = st.columns([2, 1])

with action_col1:
    if recommendations:
        for i, rec in enumerate(recommendations[:3]):
            priority_colors = ["#e74c3c", "#f39c12", "#3498db"]
            priority_labels = ["URGENT", "IMPORTANT", "RECOMMENDED"]
            icon = ["🚨", "⚠️", "💡"][min(i, 2)]
            
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, {priority_colors[i]}22, transparent); 
                        border-left: 4px solid {priority_colors[i]}; 
                        padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0;">
                <span style="background: {priority_colors[i]}; color: white; padding: 2px 8px; 
                            border-radius: 4px; font-size: 0.7rem; font-weight: bold;">
                    {priority_labels[i]}
                </span>
                <h4 style="margin: 10px 0 5px 0;">{icon} {rec.get('title', '')}</h4>
                <p style="color: #888; margin: 0;">{rec.get('description', '')}</p>
                <p style="margin: 10px 0 0 0;"><strong>Action:</strong> {rec.get('action', '')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No specific recommendations available yet.")

with action_col2:
    st.markdown("### 💰 Quick ROI Preview")
    if roi and not roi.get("error"):
        net_benefit = roi.get("net_benefit", 0)
        roi_pct = roi.get("roi_percent", 0)
        
        st.markdown(f"""
        <div style="background: #1a1a2e; padding: 20px; border-radius: 10px; text-align: center;">
            <p style="color: #888; margin: 0;">If you act on high-risk customers:</p>
            <h2 style="color: #2ecc71; margin: 10px 0;">${net_benefit:,.0f}</h2>
            <p style="color: #888; margin: 0;">potential savings</p>
            <hr style="border-color: #333; margin: 15px 0;">
            <h3 style="color: #f39c12; margin: 0;">{roi_pct:.0f}% ROI</h3>
            <p style="color: #666; font-size: 0.8rem; margin-top: 5px;">{roi.get('recommendation', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎮 Run What-If Scenarios", use_container_width=True):
            st.switch_page("pages/4_WhatIf.py")
    else:
        st.info("ROI calculation not available")

st.markdown("---")

# ====================
# RISK BREAKDOWN
# ====================
st.markdown("## 👥 Customer Risk Breakdown")

risk_col1, risk_col2 = st.columns([1, 1.5])

with risk_col1:
    if segments:
        seg_data = []
        colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#2ecc71"}
        
        for tier in ["high", "medium", "low"]:
            if tier in segments:
                seg_data.append({
                    "Risk Level": tier.upper(),
                    "Customers": segments[tier].get("count", 0),
                    "Revenue": segments[tier].get("total_clv", 0),
                    "color": colors[tier]
                })
        
        seg_df = pd.DataFrame(seg_data)
        
        fig = go.Figure(data=[go.Pie(
            labels=seg_df["Risk Level"],
            values=seg_df["Customers"],
            hole=0.6,
            marker_colors=[colors.get(t.lower(), "#999") for t in seg_df["Risk Level"]],
            textinfo='label+value',
            textposition='outside'
        )])
        
        fig.update_layout(
            title_text="Customers by Risk Level",
            showlegend=False,
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            annotations=[dict(text='Risk<br>Groups', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)

with risk_col2:
    st.markdown("### Revenue Impact by Segment")
    
    if segments:
        for tier in ["high", "medium", "low"]:
            if tier in segments:
                data = segments[tier]
                icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#2ecc71"}
                
                count = data.get('count', 0)
                total_clv = data.get('total_clv', 0)
                avg_clv = data.get('avg_clv', 0)
                churn_prob = data.get('avg_churn_prob', 0) * 100
                
                with st.expander(f"{icons[tier]} {tier.upper()} RISK — {count} customers (${total_clv:,.0f} at stake)", expanded=(tier == "high")):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Count", f"{count:,}")
                    c2.metric("Avg Value", f"${avg_clv:,.0f}")
                    c3.metric("Churn Prob", f"{churn_prob:.0f}%")
                    
                    if tier == "high":
                        st.warning("⚡ **Action needed**: These customers need immediate attention!")
                    elif tier == "medium":
                        st.info("💡 **Tip**: Engage these customers before they become high-risk")

st.markdown("---")

# ====================
# WHY CUSTOMERS LEAVE
# ====================
st.markdown("## 🔍 Why Customers Leave")
st.markdown("*Understanding these factors helps you take targeted action*")

driver_col1, driver_col2 = st.columns([1.2, 1])

with driver_col1:
    if drivers:
        # Show top 5 drivers with insights
        for i, driver in enumerate(drivers[:5]):
            feature = driver.get("feature", "").replace("_", " ").title()
            insight = driver.get("insight", "")
            importance = driver.get("importance", 0)
            
            # Progress bar color based on importance
            bar_color = "#e74c3c" if importance > 15 else "#f39c12" if importance > 10 else "#3498db"
            
            st.markdown(f"""
            <div style="margin: 10px 0; padding: 10px; background: #1a1a2e; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong>#{i+1} {feature}</strong>
                    <span style="color: {bar_color}; font-weight: bold;">{importance:.1f}%</span>
                </div>
                <div style="background: #2d2d44; height: 8px; border-radius: 4px; margin: 8px 0;">
                    <div style="background: {bar_color}; height: 100%; width: {min(importance * 3, 100)}%; border-radius: 4px;"></div>
                </div>
                <p style="color: #888; font-size: 0.85rem; margin: 0;">{insight}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No driver analysis available")

with driver_col2:
    st.markdown("### 📊 Impact Chart")
    
    if feature_importance:
        fi_df = pd.DataFrame(feature_importance[:8])
        fi_df["feature"] = fi_df["feature"].str.replace("_", " ").str.title()
        fi_df["importance"] = fi_df["importance"] * 100
        
        fig = px.bar(
            fi_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#3498db", "#e74c3c"],
        )
        fig.update_layout(
            height=350,
            showlegend=False,
            xaxis_title="Impact on Churn (%)",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ====================
# QUICK NAVIGATION
# ====================
st.markdown("## 🚀 What's Next?")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
    <div style="background: #1a1a2e; padding: 20px; border-radius: 10px; text-align: center; height: 150px;">
        <h3 style="margin-bottom: 10px;">🔍 Explore Customers</h3>
        <p style="color: #888;">Browse individual customers, see their risk scores, and understand why they might leave.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Customer List", use_container_width=True):
        st.switch_page("pages/3_Explorer.py")

with nav_col2:
    st.markdown("""
    <div style="background: #1a1a2e; padding: 20px; border-radius: 10px; text-align: center; height: 150px;">
        <h3 style="margin-bottom: 10px;">🎮 What-If Simulator</h3>
        <p style="color: #888;">Test different campaign strategies and see projected ROI before spending money.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Run Simulations", use_container_width=True):
        st.switch_page("pages/4_WhatIf.py")

with nav_col3:
    st.markdown("""
    <div style="background: #1a1a2e; padding: 20px; border-radius: 10px; text-align: center; height: 150px;">
        <h3 style="margin-bottom: 10px;">📤 New Analysis</h3>
        <p style="color: #888;">Upload new data to refresh your predictions and insights.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Upload New Data", use_container_width=True):
        del st.session_state.session_id
        st.switch_page("pages/1_Upload.py")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Quick Stats")
    st.metric("High Risk", f"{summary.get('high_risk_count', 0)} customers")
    st.metric("Revenue at Risk", f"${summary.get('revenue_at_risk', 0):,.0f}")
    st.metric("Churn Rate", f"{summary.get('churn_rate', 0) * 100:.1f}%")
    
    st.divider()
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
