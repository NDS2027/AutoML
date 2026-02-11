"""Streamlit Frontend for AutoML Advisor."""

import streamlit as st
import requests
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="AutoML Advisor - Retail Churn Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
API_URL = "http://localhost:8000/api"


def check_backend():
    """Check if backend is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #718096;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🔮 AutoML Advisor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Retail Customer Churn Prediction & Business Insights Platform</p>', unsafe_allow_html=True)
    
    # Check backend connection
    backend_status = check_backend()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/prediction.png", width=60)
        st.title("Navigation")
        
        if backend_status:
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Offline")
            st.info("Start the backend with:\n```\ncd backend\npython app.py\n```")
        
        st.divider()
        
        # Navigation info
        st.markdown("""
        ### Pages
        - **📤 Upload** - Upload your data
        - **📊 Dashboard** - View insights
        - **👥 Explorer** - Customer details
        - **🎯 What-If** - Scenario planning
        """)
        
        st.divider()
        
        # Session info
        if "session_id" in st.session_state:
            st.success(f"Session: {st.session_state.session_id[:8]}...")
        else:
            st.info("No active session")
    
    # Main content
    if not backend_status:
        st.warning("⚠️ Backend server is not running. Please start it first.")
        
        st.markdown("""
        ### How to Start
        
        1. Open a new terminal
        2. Navigate to the backend folder
        3. Run the FastAPI server:
        
        ```bash
        cd backend
        python app.py
        ```
        
        The server will start on `http://localhost:8000`
        """)
        
        if st.button("🔄 Retry Connection"):
            st.rerun()
            
        return
    
    # Welcome content
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📤 Step 1: Upload Data
        Upload your customer transaction data (CSV or Excel) to get started.
        
        **Required columns:**
        - Customer ID
        - Purchase Date
        - Transaction Amount
        """)
        
    with col2:
        st.markdown("""
        ### 🤖 Step 2: Train Models
        Our AutoML engine will automatically:
        - Clean and validate your data
        - Engineer predictive features
        - Train 5 ML models
        - Select the best performer
        """)
        
    with col3:
        st.markdown("""
        ### 📈 Step 3: Get Insights
        View comprehensive results:
        - Customer churn predictions
        - Risk segmentation
        - ROI recommendations
        - What-if simulations
        """)
    
    st.markdown("---")
    
    # Quick start
    st.markdown("### 🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Go to Upload Page", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Upload.py")
            
    with col2:
        if "session_id" in st.session_state:
            if st.button("📊 View Dashboard", use_container_width=True):
                st.switch_page("pages/2_Dashboard.py")
        else:
            st.button("📊 View Dashboard", use_container_width=True, disabled=True)
            st.caption("Upload data first to view dashboard")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #718096;'>
        <p>Built with ❤️ using Streamlit, FastAPI, and scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
