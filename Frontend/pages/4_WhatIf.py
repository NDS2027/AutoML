"""What-If Simulator Page."""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="What-If Simulator - AutoML Advisor",
    page_icon="🎯",
    layout="wide"
)

API_URL = "http://localhost:8000/api"

st.title("🎯 What-If Simulator")
st.markdown("Plan retention campaigns and calculate expected ROI.")

# Check for session
if "session_id" not in st.session_state:
    st.warning("⚠️ No analysis session found. Please upload data first.")
    if st.button("📤 Go to Upload"):
        st.switch_page("pages/1_Upload.py")
    st.stop()

session_id = st.session_state.session_id

# Campaign Parameters
st.markdown("### ⚙️ Campaign Parameters")

param_col1, param_col2, param_col3 = st.columns(3)

with param_col1:
    target_count = st.slider(
        "Number of Customers to Target",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="How many high-risk customers to include in the campaign"
    )

with param_col2:
    discount_percent = st.slider(
        "Discount Percentage (%)",
        min_value=5,
        max_value=40,
        value=20,
        step=5,
        help="Discount to offer customers"
    )

with param_col3:
    success_rate = st.slider(
        "Expected Success Rate (%)",
        min_value=10,
        max_value=60,
        value=35,
        step=5,
        help="Expected percentage of targeted customers who will return"
    ) / 100

# Run simulation
if st.button("🔮 Calculate ROI", type="primary", use_container_width=True):
    with st.spinner("Running simulation..."):
        try:
            response = requests.post(
                f"{API_URL}/simulate",
                json={
                    "session_id": session_id,
                    "target_count": target_count,
                    "discount_percent": discount_percent,
                    "success_rate": success_rate
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.simulation_result = result
            else:
                st.error("Simulation failed. Please try again.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Display results
if "simulation_result" in st.session_state:
    result = st.session_state.simulation_result
    outputs = result.get("outputs", {})
    
    if outputs.get("error"):
        st.error(outputs.get("error"))
    else:
        st.markdown("---")
        st.markdown("### 📊 Simulation Results")
        
        # Key metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                "Targeted Customers",
                f"{outputs.get('target_count', 0):,}"
            )
        
        with metric_col2:
            st.metric(
                "Total Campaign Cost",
                f"${outputs.get('total_cost', 0):,.0f}"
            )
        
        with metric_col3:
            st.metric(
                "Expected Customers Saved",
                f"{outputs.get('expected_saves', 0):,}"
            )
        
        with metric_col4:
            roi = outputs.get("roi_percent", 0)
            color = "normal" if roi > 100 else "inverse"
            st.metric(
                "Projected ROI",
                f"{roi:.0f}%",
                delta=outputs.get("recommendation", ""),
                delta_color=color
            )
        
        # Detailed breakdown
        st.markdown("---")
        st.markdown("### 💰 Financial Breakdown")
        
        breakdown_col1, breakdown_col2 = st.columns(2)
        
        with breakdown_col1:
            st.markdown("#### Costs")
            
            cost_data = {
                "Category": ["Discount Cost", "Campaign Cost", "Total Cost"],
                "Amount": [
                    outputs.get("discount_cost", 0),
                    outputs.get("campaign_cost", 0),
                    outputs.get("total_cost", 0)
                ]
            }
            
            for cat, amt in zip(cost_data["Category"], cost_data["Amount"]):
                st.markdown(f"**{cat}:** ${amt:,.0f}")
        
        with breakdown_col2:
            st.markdown("#### Revenue Impact")
            
            st.markdown(f"**CLV at Risk:** ${outputs.get('total_clv_at_risk', 0):,.0f}")
            st.markdown(f"**Avg CLV per Customer:** ${outputs.get('avg_clv', 0):,.0f}")
            st.markdown(f"**Saved Revenue:** ${outputs.get('saved_revenue', 0):,.0f}")
            st.markdown(f"**Net Benefit:** ${outputs.get('net_benefit', 0):,.0f}")
        
        # Visualization
        st.markdown("---")
        st.markdown("### 📈 ROI Visualization")
        
        # Waterfall chart
        fig = go.Figure(go.Waterfall(
            name="ROI Breakdown",
            orientation="v",
            measure=["relative", "relative", "total", "relative", "total"],
            x=["Campaign Cost", "Discount Cost", "Total Cost", "Saved Revenue", "Net Benefit"],
            y=[
                -outputs.get("campaign_cost", 0),
                -outputs.get("discount_cost", 0),
                0,  # Total calculated automatically
                outputs.get("saved_revenue", 0),
                0   # Total calculated automatically
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#3498db"}}
        ))
        
        fig.update_layout(
            title="Campaign Financial Impact",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        st.markdown("---")
        
        roi = outputs.get("roi_percent", 0)
        
        if roi >= 500:
            st.success(f"""
            ### ✅ Excellent ROI - Highly Recommended!
            
            This campaign is projected to generate a **{roi:.0f}% return on investment**.
            
            **Recommended Action:** Proceed with the campaign targeting {outputs.get('target_count', 0)} customers.
            """)
        elif roi >= 200:
            st.success(f"""
            ### ✅ Good ROI - Recommended
            
            This campaign is projected to generate a **{roi:.0f}% return on investment**.
            
            **Recommended Action:** Consider proceeding with the campaign.
            """)
        elif roi >= 100:
            st.warning(f"""
            ### ⚠️ Moderate ROI - Proceed with Caution
            
            This campaign is projected to generate a **{roi:.0f}% return on investment**.
            
            **Recommended Action:** Consider adjusting parameters to improve ROI.
            """)
        else:
            st.error(f"""
            ### ❌ Low ROI - Not Recommended
            
            This campaign is projected to generate only a **{roi:.0f}% return on investment**.
            
            **Recommended Action:** Adjust parameters or reconsider the campaign.
            """)

# Scenario comparison
st.markdown("---")
st.markdown("### 📊 Quick Scenario Comparison")

st.markdown("Compare different campaign configurations:")

scenarios = [
    {"name": "Conservative", "target": 50, "discount": 15, "success": 0.35},
    {"name": "Balanced", "target": 100, "discount": 20, "success": 0.35},
    {"name": "Aggressive", "target": 200, "discount": 25, "success": 0.35},
]

comparison_data = []

for scenario in scenarios:
    try:
        response = requests.post(
            f"{API_URL}/simulate",
            json={
                "session_id": session_id,
                "target_count": scenario["target"],
                "discount_percent": scenario["discount"],
                "success_rate": scenario["success"]
            },
            timeout=5
        )
        
        if response.status_code == 200:
            outputs = response.json().get("outputs", {})
            comparison_data.append({
                "Scenario": scenario["name"],
                "Targets": scenario["target"],
                "Discount": f"{scenario['discount']}%",
                "Cost": f"${outputs.get('total_cost', 0):,.0f}",
                "Saves": outputs.get("expected_saves", 0),
                "Revenue Saved": f"${outputs.get('saved_revenue', 0):,.0f}",
                "ROI": f"{outputs.get('roi_percent', 0):.0f}%"
            })
    except:
        pass

if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Sidebar
with st.sidebar:
    st.markdown("### 💡 Tips")
    st.markdown("""
    **Target Count:**
    Start with your highest-value at-risk customers.
    
    **Discount:**
    15-25% typically works well for retail.
    
    **Success Rate:**
    30-40% is typical for retention campaigns.
    
    **ROI Interpretation:**
    - 500%+ = Excellent
    - 200-500% = Good
    - 100-200% = Moderate
    - <100% = Consider adjusting
    """)
    
    st.divider()
    
    if st.button("📊 View Dashboard"):
        st.switch_page("pages/2_Dashboard.py")
