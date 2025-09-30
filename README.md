# 🚀 Predictive Maintenance with MLOps

This repository implements a simple pipeline for **predicting industrial equipment failures** using Random Forest and Poisson Distribution.  
It showcases how to prepare data, train models, save artifacts, evaluate metrics, and expose reusable utilities — all essential steps in an **MLOps** workflow.

---


## 🚀 How to use

 1. **Clone the repository and install dependencies:**
 
    ```bash
    git clone https://github.com/higorfct/Predictive-Maintenance-Project
    cd Predictive-Maintenance-Project
    pip install -r requirements.txt
    ```



---

## 📂 Project Structure

- **`train.py`**  
  - Loads the dataset `data/failure_dataset.csv`.
  - Performs preprocessing (standardization with `StandardScaler`).
  - Trains a baseline `RandomForestRegressor` model.
  - Runs **hyperparameter tuning** with `GridSearchCV`.
  - Saves artifacts (`model.pkl` and `scaler.pkl`) for later use.

---

## ⚙️ Pipeline Workflow (MLOps)

This project is structured to reflect basic **MLOps** principles:

1. **Versioning and Reproducibility**  
   - Separate scripts for training and evaluation.
   - Artifacts saved to disk (`model.pkl`, `scaler.pkl`), enabling version control.

2. **Automation**  
   - The flows can be integrated into CI/CD pipelines to automate retraining and monitoring.

3. **Continuous Monitoring**  
   - Evaluation metrics (MAE) allow tracking model performance over time.

---

## 💡 Business Benefits of Applying MLOps Here

Implementing **MLOps practices** in this predictive maintenance scenario delivers clear business value:

- **Reduced Downtime Costs:**  
  Early failure prediction helps plan maintenance proactively, minimizing unplanned outages.

- **Faster Iteration & Deployment:**  
  Automated training, evaluation, and artifact management shorten time-to-market for improved models.

- **Consistency & Quality Assurance:**  
  Versioned artifacts and reproducible pipelines reduce errors and ensure stable model performance in production.

- **Scalable Decision-Making:**  
  Utility functions can be embedded into production systems, enabling real-time risk assessment across multiple assets.

- **Operational Efficiency & Savings:**  
  Less manual intervention in model training and deployment translates to lower operational costs and more reliable insights for maintenance teams.

In short, MLOps transforms this model from a one-off experiment into a **scalable, maintainable, and business-impacting solution**.

---
