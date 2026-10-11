import pandas as pd
import numpy as np

def generate_synthetic_market_data(days=365):
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=days)
    regions = ['Market_A', 'Market_B', 'Market_C', 'Market_D'] 
    
    data = []
    for region in regions:
        base_demand = np.random.normal(1000, 100, days)
        marketing_spend = np.random.normal(5000, 500, days)
        competitor_price = np.random.normal(50, 2, days)
        
        # Treatment: Price change in Market_A at day 200
        if region == 'Market_A':
            price = np.where(np.arange(days) >= 200, 45, 50) 
            is_treated = np.where(np.arange(days) >= 200, 1, 0)
        else:
            price = np.random.normal(50, 1, days)
            is_treated = np.zeros(days)
            
        revenue = (base_demand * 10) - (price * 50) + (competitor_price * 20) + (marketing_spend * 0.5) + np.random.normal(0, 500, days)
        
        df = pd.DataFrame({
            'date': dates,
            'region': region,
            'price': price,
            'competitor_price': competitor_price,
            'marketing_spend': marketing_spend,
            'revenue': revenue,
            'is_treated': is_treated
        })
        data.append(df)

    final_df = pd.concat(data, ignore_index=True)
    final_df.to_csv('data/market_telemetry.csv', index=False)
    print("Synthetic dataset generated at data/market_telemetry.csv")

if __name__ == "__main__":
    generate_synthetic_market_data()