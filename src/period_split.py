import pandas as pd

def split_treatment_periods():
    df = pd.read_csv('data/market_telemetry.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Pivot to align regions side-by-side for the synthetic control
    pivot_df = df.pivot(index='date', columns='region', values='revenue').dropna()
    
    # Treatment occurred on day 200
    treatment_date = pivot_df.index[200]
    
    pre_treatment = pivot_df[pivot_df.index < treatment_date]
    post_treatment = pivot_df[pivot_df.index >= treatment_date]
    
    print(f"Day 6: Periods separated. Pre-treatment shape: {pre_treatment.shape}, Post-treatment shape: {post_treatment.shape}")
    
    # Save the split index for downstream tasks
    pre_treatment.to_csv('data/pre_treatment.csv')
    post_treatment.to_csv('data/post_treatment.csv')

if __name__ == "__main__":
    split_treatment_periods()