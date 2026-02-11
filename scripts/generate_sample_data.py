"""Generate sample retail transaction data for testing."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(
    num_customers: int = 500,
    num_transactions: int = 5000,
    output_path: str = "sample_retail_data.csv"
):
    """
    Generate synthetic retail transaction data.
    
    Args:
        num_customers: Number of unique customers
        num_transactions: Total number of transactions
        output_path: Path to save the CSV
    """
    
    np.random.seed(42)
    random.seed(42)
    
    # Customer IDs
    customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, num_customers + 1)]
    
    # Products and categories
    products = {
        "Electronics": ["Laptop", "Phone", "Headphones", "Tablet", "Smartwatch"],
        "Clothing": ["T-Shirt", "Jeans", "Jacket", "Shoes", "Hat"],
        "Home": ["Lamp", "Pillow", "Rug", "Vase", "Frame"],
        "Sports": ["Yoga Mat", "Dumbbells", "Running Shoes", "Water Bottle", "Gym Bag"],
        "Books": ["Novel", "Textbook", "Magazine", "Notebook", "Planner"]
    }
    
    # Generate base date range (last 18 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=540)
    
    transactions = []
    
    # Assign customer profiles
    customer_profiles = {}
    for cust_id in customer_ids:
        # Random profile type
        profile = random.choices(
            ["loyal", "regular", "occasional", "churning", "new"],
            weights=[0.15, 0.30, 0.25, 0.15, 0.15]
        )[0]
        
        customer_profiles[cust_id] = {
            "profile": profile,
            "avg_spend": random.uniform(20, 200),
            "favorite_category": random.choice(list(products.keys())),
            "first_purchase": start_date + timedelta(days=random.randint(0, 400))
        }
    
    # Generate transactions
    for _ in range(num_transactions):
        cust_id = random.choice(customer_ids)
        profile = customer_profiles[cust_id]
        
        # Generate date based on profile
        if profile["profile"] == "loyal":
            # Regular purchases throughout
            days_offset = random.randint(0, 540)
        elif profile["profile"] == "regular":
            days_offset = random.randint(0, 540)
        elif profile["profile"] == "occasional":
            days_offset = random.randint(0, 540)
        elif profile["profile"] == "churning":
            # Most purchases were early, few recent
            days_offset = random.choices(
                range(540),
                weights=[max(0.1, (540 - i) / 540) for i in range(540)]
            )[0]
        else:  # new
            # Recent purchases only
            days_offset = random.randint(0, 120)
        
        purchase_date = start_date + timedelta(days=days_offset)
        
        # Ensure after first purchase
        if purchase_date < profile["first_purchase"]:
            purchase_date = profile["first_purchase"] + timedelta(days=random.randint(0, 30))
        
        # Generate amount
        base_amount = profile["avg_spend"]
        amount = max(5, np.random.normal(base_amount, base_amount * 0.3))
        
        # Select product
        category = random.choices(
            list(products.keys()),
            weights=[0.4 if c == profile["favorite_category"] else 0.15 for c in products.keys()]
        )[0]
        product = random.choice(products[category])
        
        transactions.append({
            "customer_id": cust_id,
            "transaction_date": purchase_date.strftime("%Y-%m-%d"),
            "amount": round(amount, 2),
            "product_name": product,
            "category": category
        })
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Sort by date
    df = df.sort_values("transaction_date")
    
    # Add transaction ID
    df.insert(0, "transaction_id", [f"TXN{str(i).zfill(8)}" for i in range(1, len(df) + 1)])
    
    # Save
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} transactions for {num_customers} customers")
    print(f"Saved to: {output_path}")
    
    # Print summary
    print("\nData Summary:")
    print(f"  Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"  Total revenue: ${df['amount'].sum():,.2f}")
    print(f"  Avg order value: ${df['amount'].mean():.2f}")
    print(f"  Unique customers: {df['customer_id'].nunique()}")
    
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Generate to samples folder
    output_dir = Path(__file__).parent.parent / "data" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "sample_retail_data.csv"
    
    generate_sample_data(
        num_customers=500,
        num_transactions=5000,
        output_path=str(output_path)
    )
