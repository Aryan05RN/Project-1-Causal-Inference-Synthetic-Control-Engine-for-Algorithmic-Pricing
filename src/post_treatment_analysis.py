
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

def analyze_post_treatment():
    # Load pre-treatment and post-treatment data
    pre = pd.read_csv("data/pre_treatment.csv", index_col="date")
    post = pd.read_csv("data/post_treatment.csv", index_col="date")

    # Load the trained synthetic control model
    model = joblib.load("data/synthetic_control_model.pkl")

    donor_columns = ["Market_B", "Market_C", "Market_D"]

    # Predict Market A's counterfactual revenue
    # (estimated revenue without the treatment)
    synthetic_pre = model.predict(pre[donor_columns])
    synthetic_post = model.predict(post[donor_columns])

    # Pre-treatment fit diagnostic
    pre_mspe = mean_squared_error(
        pre["Market_A"], synthetic_pre
    )

    # Actual versus counterfactual after treatment
    actual_post = post["Market_A"].to_numpy()
    effect = actual_post - synthetic_post

    average_effect = np.mean(effect)
    total_effect = np.sum(effect)

    print("\n=== POST-TREATMENT IMPACT ANALYSIS ===")
    print(f"Post-treatment observations: {len(post)}")
    print(f"Pre-treatment MSPE: {pre_mspe:.2f}")
    print(f"Average daily revenue effect: {average_effect:.2f}")
    print(f"Estimated cumulative revenue effect: {total_effect:.2f}")

    # Save detailed results
    results = pd.DataFrame({
        "date": post.index,
        "actual_revenue": actual_post,
        "synthetic_revenue": synthetic_post,
        "estimated_effect": effect
    })

    results.to_csv(
        "data/post_treatment_effects.csv", index=False
    )

    print("\nResults saved to data/post_treatment_effects.csv")

if __name__ == "__main__":
    analyze_post_treatment()
