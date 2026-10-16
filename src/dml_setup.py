import pandas as pd
import numpy as np

def prepare_dml_tensors():
    df = pd.read_csv('data/market_telemetry.csv')
    
    # Focus on Market A for DML treatment effect estimation
    df_a = df[df['region'] == 'Market_A'].copy()
    
    # Define DML vectors
    Y = df_a['revenue'].values  # Outcome
    T = df_a['is_treated'].values  # Treatment
    X = df_a[['competitor_price', 'marketing_spend']].values  # High-dimensional Confounders
    
    # Save tensors
    np.savez('data/dml_tensors.npz', Y=Y, T=T, X=X)
    print(f"Day 9: DML tensors prepared. Y shape: {Y.shape}, X shape: {X.shape}")

if __name__ == "__main__":
    prepare_dml_tensors()