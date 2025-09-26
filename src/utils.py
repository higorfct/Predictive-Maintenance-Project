import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
import joblib
import numpy as np

def predict_failure(user_input: list):
    """Receives [operation_time, temperature, vibration] and returns the predicted value of lambda"""
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    df_input = pd.DataFrame([user_input], columns=["operation_time", "temperature", "vibration"])
    scaled = scaler.transform(df_input)
    return model.predict(scaled)[0]

def plot_poisson_distribution(lambda_predicted: float):
    """Plots poisson distribution based on predicted λ """
    x_vals = np.arange(0, int(lambda_predicted + 3))
    probs = poisson.pmf(x_vals, lambda_predicted)
    labels = [f"{i}: {round(p, 4)}" for i, p in zip(x_vals, probs)]
    theme_colors = [plt.cm.Greys(0.2 + 0.6 * (p / max(probs))) for p in probs]
    
    fig, ax = plt.subplots()
    ax.bar(x_vals, probs, tick_label=labels, color=theme_colors)
    ax.set_title(f"Poisson Distribution (λ={lambda_predicted:.2f})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
