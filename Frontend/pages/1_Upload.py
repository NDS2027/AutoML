"""Upload and Configure Page."""

import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(
    page_title="Upload Data - AutoML Advisor",
    page_icon="📤",
    layout="wide"
)

API_URL = "http://localhost:8000/api"

st.title("📤 Upload & Configure")
st.markdown("Upload your customer transaction data to begin churn analysis.")

# Initialize session state
if "upload_id" not in st.session_state:
    st.session_state.upload_id = None
if "columns" not in st.session_state:
    st.session_state.columns = []
if "preview" not in st.session_state:
    st.session_state.preview = None

# Step 1: Upload File
st.markdown("### Step 1: Upload Your Data")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Upload customer transaction data with columns for Customer ID, Date, and Amount"
)

if uploaded_file is not None:
    with st.spinner("Uploading and analyzing file..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
            response = requests.post(f"{API_URL}/upload", files=files)
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.upload_id = data["upload_id"]
                st.session_state.columns = data["columns"]
                st.session_state.preview = data["preview"]
                st.session_state.detected_types = data["detected_types"]
                
                st.success(f"✅ File uploaded successfully! Found {data['rows']:,} records.")
            else:
                st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")

# Show preview if uploaded
if st.session_state.preview:
    st.markdown("### Data Preview")
    preview_df = pd.DataFrame(st.session_state.preview)
    st.dataframe(preview_df, use_container_width=True)
    
    # Step 2: Column Mapping
    st.markdown("---")
    st.markdown("### Step 2: Map Your Columns")
    st.markdown("Select which columns in your data correspond to the required fields.")
    
    columns = st.session_state.columns
    
    # Smart column detection
    def find_column_index(columns, keywords):
        """Find column index matching any keyword (case-insensitive)."""
        for i, col in enumerate(columns):
            col_lower = col.lower()
            for kw in keywords:
                if kw in col_lower:
                    return i
        return 0
    
    # Auto-detect columns based on common names
    customer_idx = find_column_index(columns, ["customer_id", "customer", "cust", "client", "user_id", "userid"])
    date_idx = find_column_index(columns, ["date", "transaction_date", "purchase_date", "order_date", "created"])
    amount_idx = find_column_index(columns, ["amount", "total", "value", "price", "revenue", "sales"])
    product_idx = find_column_index(columns, ["product", "item", "sku", "category"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        customer_id_col = st.selectbox(
            "Customer ID Column *",
            options=columns,
            index=customer_idx,
            help="The column containing unique customer identifiers"
        )
        
        date_col = st.selectbox(
            "Transaction Date Column *",
            options=columns,
            index=date_idx,
            help="The column containing purchase dates"
        )
    
    with col2:
        amount_col = st.selectbox(
            "Amount Column *",
            options=columns,
            index=amount_idx,
            help="The column containing transaction amounts"
        )
        
        product_col = st.selectbox(
            "Product Column (Optional)",
            options=["None"] + columns,
            index=product_idx + 1 if product_idx > 0 else 0,
            help="Optional: Column containing product names or categories"
        )
    
    # Step 3: Churn Definition
    st.markdown("---")
    st.markdown("### Step 3: Define Churn Threshold")
    st.markdown("A customer is considered 'churned' if they haven't purchased within this many days.")
    
    churn_threshold = st.slider(
        "Churn Threshold (days)",
        min_value=30,
        max_value=180,
        value=60,
        step=15,
        help="Number of days without purchase to consider a customer churned"
    )
    
    st.info(f"📊 Customers who haven't purchased in **{churn_threshold}** days will be labeled as churned.")
    
    # Configure button
    st.markdown("---")
    
    if st.button("🚀 Configure & Start Analysis", type="primary", use_container_width=True):
        with st.spinner("Configuring analysis..."):
            try:
                # Send configuration
                config_data = {
                    "upload_id": st.session_state.upload_id,
                    "column_mapping": {
                        "customer_id": customer_id_col,
                        "date": date_col,
                        "amount": amount_col,
                        "product": product_col if product_col != "None" else None
                    },
                    "churn_threshold_days": churn_threshold
                }
                
                response = requests.post(f"{API_URL}/configure", json=config_data)
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.session_id = data["session_id"]
                    
                    # Show validation results
                    validation = data.get("validation_results", {})
                    
                    if validation.get("is_valid", True):
                        st.success("✅ Configuration successful! Data validated.")
                        
                        # Start training
                        st.markdown("---")
                        st.markdown("### 🤖 Training Models")
                        
                        train_response = requests.post(
                            f"{API_URL}/train",
                            json={"session_id": st.session_state.session_id}
                        )
                        
                        if train_response.status_code == 200:
                            job_data = train_response.json()
                            job_id = job_data["job_id"]
                            
                            # Progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Poll for status
                            while True:
                                status_response = requests.get(f"{API_URL}/train/status/{job_id}")
                                
                                if status_response.status_code == 200:
                                    status = status_response.json()
                                    progress = status.get("progress", 0)
                                    current_step = status.get("current_step", "Processing...")
                                    
                                    progress_bar.progress(progress / 100)
                                    status_text.text(f"Step: {current_step} ({progress}%)")
                                    
                                    if status.get("status") == "completed":
                                        st.success("✅ Training complete! Models are ready.")
                                        st.balloons()
                                        
                                        if st.button("📊 View Dashboard", type="primary"):
                                            st.switch_page("pages/2_Dashboard.py")
                                        break
                                        
                                    elif status.get("status") == "failed":
                                        st.error(f"Training failed: {status.get('error', 'Unknown error')}")
                                        break
                                
                                time.sleep(2)
                        else:
                            st.error("Failed to start training")
                    else:
                        st.error("Data validation failed:")
                        for issue in validation.get("issues", []):
                            st.warning(f"• {issue}")
                else:
                    st.error(f"Configuration failed: {response.json().get('detail', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Sidebar info
with st.sidebar:
    st.markdown("### 📋 Data Requirements")
    st.markdown("""
    Your data should contain:
    - **Customer ID**: Unique identifier for each customer
    - **Transaction Date**: When each purchase was made
    - **Amount**: Value of each transaction
    - **Product** (optional): Product name or category
    
    **Minimum Requirements:**
    - At least 500 transactions
    - At least 100 unique customers
    - Data spanning 6+ months
    """)
    
    if st.session_state.upload_id:
        st.success(f"Upload ID: {st.session_state.upload_id[:8]}...")
    
    if "session_id" in st.session_state:
        st.success(f"Session: {st.session_state.session_id[:8]}...")
