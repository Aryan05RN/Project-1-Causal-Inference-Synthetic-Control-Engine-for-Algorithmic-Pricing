import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

def run_pure_sklearn_dml():
    # 1. Load the structured confounder tensors
    tensors = np.load('data/dml_tensors.npz')
    Y, T, X = tensors['Y'], tensors['T'], tensors['X']
    
    print("Stage 1: Isolating confounders using Machine Learning...")
    # 2. Train models to predict Treatment (T) and Outcome (Y) strictly from Confounders (X)
    # We use cross_val_predict to generate unbiased, out-of-fold predictions
    model_t = HistGradientBoostingClassifier(random_state=42)
    model_y = HistGradientBoostingRegressor(random_state=42)
    
    T_pred = cross_val_predict(model_t, X, T, cv=3, method='predict_proba')[:, 1]
    Y_pred = cross_val_predict(model_y, X, Y, cv=3)
    
    # 3. Calculate Residuals (Orthogonalization)
    # This removes the "spillover" effects of the confounders
    T_res = T - T_pred
    Y_res = Y - Y_pred
    
    print("Stage 2: Estimating Causal Lift...")
    # 4. Regress the Outcome residuals on the Treatment residuals to find the true effect
    effect_model = LinearRegression()
    effect_model.fit(T_res.reshape(-1, 1), Y_res)
    
    ate = effect_model.coef_[0]
    print("DML Causal Lift Estimator active.")
    print(f"Estimated Revenue Treatment Effect: {ate:.2f}")

if __name__ == "__main__":
    run_pure_sklearn_dml()