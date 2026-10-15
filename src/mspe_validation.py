import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

def calculate_mspe():
    pre_treatment = pd.read_csv('data/pre_treatment.csv', index_col='date')
    X_donor = pre_treatment[['Market_B', 'Market_C', 'Market_D']]
    y_actual = pre_treatment['Market_A']
    
    # Load optimized model
    model = joblib.load('data/synthetic_control_model.pkl')
    
    # Generate predictions
    y_synthetic = model.predict(X_donor)
    
    # Calculate MSPE
    mspe = mean_squared_error(y_actual, y_synthetic)
    print(f"Day 8: Validation complete. Pre-treatment MSPE: {mspe:.2f}")

if __name__ == "__main__":
    calculate_mspe()