import pandas as pd
from dowhy import CausalModel
import matplotlib.pyplot as plt

def build_causal_graph():
    df = pd.read_csv('data/market_telemetry.csv')
    
    causal_graph = """
    digraph {
        competitor_price -> revenue;
        competitor_price -> price;
        marketing_spend -> revenue;
        price -> revenue;
        
        price[label="Price (Treatment)"];
        revenue[label="Revenue (Outcome)"];
        competitor_price[label="Competitor Price (Confounder)"];
        marketing_spend[label="Marketing Spend (Confounder)"];
    }
    """
    
    model = CausalModel(
        data=df,
        treatment='price',
        outcome='revenue',
        graph=causal_graph
    )
    
    model.view_model(layout="dot")
    print("Causal DAG successfully constructed and mapped.")

if __name__ == "__main__":
    build_causal_graph()