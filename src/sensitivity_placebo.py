import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

def run_placebo_test():
    df = pd.read_csv('data/market_telemetry.csv')
    
    # Use Market_B (an untreated donor unit) as a placebo
    df_placebo = df[df['region'] == 'Market_B'].copy()
    
    # Inject fake treatment on day 200
    df_placebo['fake_treatment'] = np.where(df_placebo.index >= 200, 1, 0)
    
    Y = df_placebo['revenue'].values
    T = df_placebo['fake_treatment'].values
    X = df_placebo[['competitor_price', 'marketing_spend']].values
    
    # Orthogonalization
    T_pred = cross_val_predict(HistGradientBoostingClassifier(random_state=42), X, T, cv=3, method='predict_proba')[:, 1]
    Y_pred = cross_val_predict(HistGradientBoostingRegressor(random_state=42), X, Y, cv=3)
    
    T_res = T - T_pred
    Y_res = Y - Y_pred
    
    ate_model = LinearRegression().fit(T_res.reshape(-1, 1), Y_res)
    
    print("Sensitivity Analysis: Placebo Treatment Test")
    print(f"Placebo Treatment Effect (Should be close to 0): {ate_model.coef_[0]:.2f}")
    if abs(ate_model.coef_[0]) < 150:
        print("Robustness Confirmed: No significant causal lift detected on untreated placebo unit.")

if __name__ == "__main__":
    run_placebo_test()