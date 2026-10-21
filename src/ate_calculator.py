import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

def calculate_ate():
    tensors = np.load('data/dml_tensors.npz')
    Y, T, X = tensors['Y'], tensors['T'], tensors['X']
    
    # Orthogonalization
    model_t = HistGradientBoostingClassifier(random_state=42)
    model_y = HistGradientBoostingRegressor(random_state=42)
    
    T_pred = cross_val_predict(model_t, X, T, cv=3, method='predict_proba')[:, 1]
    Y_pred = cross_val_predict(model_y, X, Y, cv=3)
    
    T_res = T - T_pred
    Y_res = Y - Y_pred
    
    # ATE Calculation
    ate_model = LinearRegression()
    ate_model.fit(T_res.reshape(-1, 1), Y_res)
    ate = ate_model.coef_[0]
    
    print(f"Global Average Treatment Effect (ATE) successfully quantified.")
    print(f"Revenue Impact per Treated Unit: {ate:.2f}")

if __name__ == "__main__":
    calculate_ate()