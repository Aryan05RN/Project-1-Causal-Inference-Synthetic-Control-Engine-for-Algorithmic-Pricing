# Causal Inference & Synthetic Control Engine for Algorithmic Pricing

## 1. Project Overview

This project aims to estimate the causal impact of dynamic pricing changes on revenue in interconnected markets.

Traditional A/B testing may not fully capture market spillovers. This project explores causal inference methods to estimate what revenue might have been if a pricing intervention had not occurred.

## 2. Objectives

- Analyze regional market transaction data and competitor pricing.
- Represent causal assumptions using Directed Acyclic Graphs (DAGs).
- Separate pre-treatment and post-treatment periods.
- Estimate counterfactual outcomes using Synthetic Control.
- Apply Double Machine Learning (DML) to estimate treatment effects.
- Validate pre-treatment fit using Mean Squared Prediction Error (MSPE).
- Develop an interactive pricing analysis dashboard.

## 3. Technologies

- Python
- Pandas and NumPy
- DoWhy
- Scikit-learn
- EconML
- Statsmodels
- Streamlit or Dash
- Git and GitHub

*Dependencies and versions will be confirmed against the implementation.*

## 4. Repository Structure

```text
├── data/
│   ├── market_telemetry.csv
│   ├── pre_treatment.csv
│   ├── post_treatment.csv
│   └── dml_tensors.npz
├── src/
│   ├── causal_dag.py
│   ├── dml_estimator.py
│   ├── dml_setup.py
│   ├── generate_data.py
│   ├── mspe_validation.py
│   ├── period_split.py
│   └── synthetic_control.py
├── causal_model.png
└── README.md
```

## 5. Project Status

The repository contains initial data files and Python modules for causal analysis, Synthetic Control, DML, and model validation.

The correctness and execution status of these modules still need to be verified.

## 6. Getting Started

1. Install a compatible Python version.
2. Create and activate a virtual environment.
3. Install the project's confirmed dependencies.
4. Follow the team's execution instructions once they are documented.

## 7. Team Collaboration

This is a collaborative internship project. Team members should use meaningful commit messages, coordinate changes, and follow the repository's agreed branching and review process.

## 8. Results

Validated results, model diagnostics, and causal effect estimates will be added after the implementation and evaluation are complete.