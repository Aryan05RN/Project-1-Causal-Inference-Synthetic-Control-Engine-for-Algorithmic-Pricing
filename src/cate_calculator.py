import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

def calculate_cate():
    tensors = np.load('data/dml_tensors.npz')
    Y, T, X = tensors['Y'], tensors['T'], tensors['X']
    
    T_pred = cross_val_predict(HistGradientBoostingClassifier(random_state=42), X, T, cv=3, method='predict_proba')[:, 1]
    Y_pred = cross_val_predict(HistGradientBoostingRegressor(random_state=42), X, Y, cv=3)
    
    T_res = T - T_pred
    Y_res = Y - Y_pred
    
    # Interact Treatment Residuals with Confounders (X) to find Conditional Effects
    # X[:, 0] is competitor_price, X[:, 1] is marketing_spend
    interaction_features = T_res[:, np.newaxis] * X
    
    cate_model = LinearRegression()
    cate_model.fit(interaction_features, Y_res)
    
    print("Conditional Average Treatment Effect (CATE) metrics:")
    print(f"Sensitivity to Competitor Price: {cate_model.coef_[0]:.4f}")
    print(f"Sensitivity to Marketing Spend: {cate_model.coef_[1]:.4f}")

if __name__ == "__main__":
    calculate_cate()