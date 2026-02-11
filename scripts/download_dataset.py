"""Download UCI Online Retail Dataset for testing."""

import pandas as pd
import requests
import zipfile
import io
from pathlib import Path

def download_uci_retail():
    """Download and prepare the UCI Online Retail dataset."""
    
    print("Downloading UCI Online Retail dataset...")
    
    # UCI dataset URL (Excel file)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    
    # Alternative: Kaggle mirror (CSV) - use if UCI is slow
    alt_url = "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"
    
    try:
        # Try the original UCI Excel file
        print("Trying UCI source...")
        response = requests.get(url, timeout=60)
        
        if response.status_code == 200:
            print("Downloaded successfully, processing...")
            df = pd.read_excel(io.BytesIO(response.content))
        else:
            raise Exception("UCI download failed")
            
    except Exception as e:
        print(f"UCI source failed ({e}), trying alternative...")
        try:
            response = requests.get(alt_url, timeout=60)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
            else:
                print("Both sources failed. Creating sample from description...")
                return create_sample_from_description()
        except:
            return create_sample_from_description()
    
    # Clean and prepare the dataset
    print(f"Raw data: {len(df)} rows")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Rename columns to match our expected format
    column_map = {
        'invoiceno': 'transaction_id',
        'invoice': 'transaction_id',
        'customerid': 'customer_id', 
        'customer_id': 'customer_id',
        'invoicedate': 'transaction_date',
        'invoice_date': 'transaction_date',
        'unitprice': 'unit_price',
        'description': 'product_name',
        'stockcode': 'product_code'
    }
    
    df = df.rename(columns=column_map)
    
    # Calculate amount
    if 'quantity' in df.columns and 'unit_price' in df.columns:
        df['amount'] = df['quantity'] * df['unit_price']
    elif 'amount' not in df.columns:
        df['amount'] = 50.0  # Default
    
    # Remove rows with missing customer_id
    if 'customer_id' in df.columns:
        df = df.dropna(subset=['customer_id'])
        df['customer_id'] = df['customer_id'].astype(int).astype(str)
    
    # Remove negative amounts (returns)
    df = df[df['amount'] > 0]
    
    # Select and order columns
    final_columns = ['transaction_id', 'customer_id', 'transaction_date', 'amount', 'product_name']
    available = [c for c in final_columns if c in df.columns]
    df = df[available]
    
    # Take a sample if too large
    if len(df) > 50000:
        print(f"Sampling 50,000 rows from {len(df)}...")
        df = df.sample(n=50000, random_state=42)
    
    # Save
    output_path = Path(__file__).parent / "data" / "samples" / "online_retail_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Unique customers: {df['customer_id'].nunique():,}")
    print(f"   Columns: {list(df.columns)}")
    
    return output_path


def create_sample_from_description():
    """Create a larger sample dataset if download fails."""
    import random
    from datetime import datetime, timedelta
    
    print("Creating sample retail dataset...")
    
    # Generate realistic data
    num_customers = 500
    num_transactions = 5000
    
    customers = [f"CUST{str(i).zfill(5)}" for i in range(1, num_customers + 1)]
    products = [
        "Laptop", "Phone", "Tablet", "Headphones", "Keyboard", "Mouse",
        "Monitor", "Printer", "Camera", "Speaker", "Charger", "Cable",
        "Case", "Stand", "Bag", "T-Shirt", "Jeans", "Shoes", "Watch", "Sunglasses"
    ]
    
    base_date = datetime(2024, 1, 1)
    
    transactions = []
    for i in range(num_transactions):
        customer = random.choice(customers)
        product = random.choice(products)
        days_offset = random.randint(0, 365)
        date = base_date + timedelta(days=days_offset)
        amount = round(random.uniform(10, 500), 2)
        
        transactions.append({
            "transaction_id": f"TXN{str(i+1).zfill(8)}",
            "customer_id": customer,
            "transaction_date": date.strftime("%Y-%m-%d"),
            "amount": amount,
            "product_name": product
        })
    
    df = pd.DataFrame(transactions)
    
    output_path = Path(__file__).parent / "data" / "samples" / "online_retail_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Sample dataset created: {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Unique customers: {df['customer_id'].nunique():,}")
    
    return output_path


if __name__ == "__main__":
    download_uci_retail()
