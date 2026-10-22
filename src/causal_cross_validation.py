import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

def cross_validate_causal_model():
    tensors = np.load('data/dml_tensors.npz')
    Y, T, X = tensors['Y'], tensors['T'], tensors['X']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ate_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        # Train Nuisance Models
        model_t = HistGradientBoostingClassifier(random_state=42).fit(X_train, T_train)
        model_y = HistGradientBoostingRegressor(random_state=42).fit(X_train, Y_train)
        
        # Predict on Test Fold
        T_res = T_test - model_t.predict_proba(X_test)[:, 1]
        Y_res = Y_test - model_y.predict(X_test)
        
        # Estimate Fold ATE
        ate_model = LinearRegression().fit(T_res.reshape(-1, 1), Y_res)
        ate_scores.append(ate_model.coef_[0])
        
    print("Cross-Validation techniques applied to causal models.")
    print(f"ATE Folds: {[round(score, 2) for score in ate_scores]}")
    print(f"Robust Mean ATE: {np.mean(ate_scores):.2f} (+/- {np.std(ate_scores) * 2:.2f})")

if __name__ == "__main__":
    cross_validate_causal_model()