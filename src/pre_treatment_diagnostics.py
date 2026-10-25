
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_diagnostics():
    pre = pd.read_csv("data/pre_treatment.csv", index_col="date")

    model = joblib.load("data/synthetic_control_model.pkl")
    donors = ["Market_B", "Market_C", "Market_D"]

    actual = pre["Market_A"].to_numpy()
    predicted = model.predict(pre[donors])

    errors = actual - predicted

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mean_actual = np.mean(actual)

    print("\n=== PRE-TREATMENT DIAGNOSTICS ===")
    print(f"Observations: {len(pre)}")
    print(f"Mean actual revenue: {mean_actual:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE as % of mean revenue: {rmse / mean_actual * 100:.2f}%")
    print(f"Mean prediction error: {np.mean(errors):.2f}")

    if rmse / mean_actual > 0.10:
        print("WARNING: Pre-treatment fit may need improvement.")
    else:
        print("Pre-treatment fit passes the basic 10% RMSE check.")

if __name__ == "__main__":
    run_diagnostics()
