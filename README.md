# 🚀 Predictive Maintenance with MLOps

This repository implements a simple pipeline for **predicting industrial equipment failures** using Random Forest and Poisson Distribution.  
It showcases how to prepare data, train models, save artifacts, evaluate metrics, and expose reusable utilities — all essential steps in an **MLOps** workflow.

---

## 📂 Project Structure

- **`train.py`**  
  - Loads the dataset `data/failure_dataset.csv`.
  - Performs preprocessing (standardization with `StandardScaler`).
  - Trains a baseline `RandomForestRegressor` model.
  - Runs **hyperparameter tuning** with `GridSearchCV`.
  - Saves artifacts (`model.pkl` and `scaler.pkl`) for later use.

- **`evaluate.py`**  
  - Loads the saved artifacts.
  - Makes predictions on the test set.
  - Computes performance metrics (MAE and R²) to monitor model quality.

- **`utils.py`**  
  - `predict_failure`: receives operational data and returns the model’s prediction.
  - `plot_poisson_distribution`: generates a Poisson distribution visualization from the predicted failure rate.

---

## ⚙️ Pipeline Workflow (MLOps)

This project is structured to reflect basic **MLOps** principles:

1. **Versioning and Reproducibility**  
   - Separate scripts for training and evaluation.
   - Artifacts saved to disk (`model.pkl`, `scaler.pkl`), enabling version control.

2. **Automation**  
   - The training (`train.py`) and evaluation (`evaluate.py`) flows can be integrated into CI/CD pipelines to automate retraining and monitoring.

3. **Continuous Monitoring**  
   - Evaluation metrics (MAE and R²) allow tracking model performance over time.

4. **Reuse and Deployment**  
   - The `utils.py` module abstracts functions for use in APIs, dashboards, or external scripts.

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

## 📝 How to Use

Run the whole pipeline (install dependencies, train and evaluate the model):

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Evaluate the model
python evaluate.py
