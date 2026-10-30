import pandas as pd
import numpy as np
import joblib

def simulate_counterfactual_revenue(new_price, competitor_price, marketing_spend):
    # Load historical baseline fit
    synthetic_model = joblib.load('data/synthetic_control_model.pkl')
    
    # Base baseline revenue prediction
    base_donor_values = np.array([[5000, 4800, 5100]])
    baseline_revenue = synthetic_model.predict(base_donor_values)[0]
    
    # Price elasticity lift estimation (-45 per unit price increase from baseline 50)
    price_delta = 50.0 - new_price
    causal_lift = (price_delta * 42.5) + (marketing_spend * 0.05) - (competitor_price * 10)
    
    projected_revenue = baseline_revenue + causal_lift
    return baseline_revenue, projected_revenue, causal_lift

if __name__ == "__main__":
    base, proj, lift = simulate_counterfactual_revenue(45.0, 50.0, 5000.0)
    print(f"Simulated Baseline: ${base:.2f} | Projected: ${proj:.2f} | Causal Lift: ${lift:.2f}")