import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import joblib

def optimize_synthetic_control():
    pre_treatment = pd.read_csv('data/pre_treatment.csv', index_col='date')
    
    # Market A is treated; B, C, D are the donor pool
    X_donor = pre_treatment[['Market_B', 'Market_C', 'Market_D']]
    y_treated = pre_treatment['Market_A']
    
    # Ridge regression to find optimal donor pool weights
    optimizer = Ridge(alpha=1.0, positive=True) # positive weights for synthetic control
    optimizer.fit(X_donor, y_treated)
    
    print(f"Day 7: Optimization complete. Donor Pool Weights: {optimizer.coef_}")
    
    # Save model for MSPE calculation
    joblib.dump(optimizer, 'data/synthetic_control_model.pkl')

if __name__ == "__main__":
    optimize_synthetic_control()