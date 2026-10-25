
import pandas as pd
import matplotlib.pyplot as plt

def plot_treatment_effect():
    pre = pd.read_csv("data/pre_treatment.csv")
    post = pd.read_csv("data/post_treatment_effects.csv")

    pre["date"] = pd.to_datetime(pre["date"])
    post["date"] = pd.to_datetime(post["date"])

    # Combine the actual pre-treatment observations
    # with the post-treatment actual and synthetic values
    actual_pre = pre[["date", "Market_A"]].rename(
        columns={"Market_A": "actual_revenue"}
    )
    actual_pre["synthetic_revenue"] = float("nan")

    combined = pd.concat([
        actual_pre,
        post[["date", "actual_revenue", "synthetic_revenue"]]
    ], ignore_index=True).sort_values("date")

    plt.figure(figsize=(12, 6))

    plt.plot(
        combined["date"],
        combined["actual_revenue"],
        label="Market A Actual Revenue"
    )
    plt.plot(
        combined["date"],
        combined["synthetic_revenue"],
        label="Synthetic Market A Revenue"
    )

    treatment_date = post["date"].min()
    plt.axvline(
        treatment_date,
        linestyle="--",
        label="Treatment Start"
    )

    plt.title("Actual vs Synthetic Revenue After Price Change")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("data/treatment_effect_chart.png", dpi=150)
    plt.show()

    print("Chart saved to data/treatment_effect_chart.png")

if __name__ == "__main__":
    plot_treatment_effect()
