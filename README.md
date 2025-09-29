# 🚀 Predictive Maintenance with MLOps  

This repository implements a pipeline for **predicting industrial equipment failures** using Random Forest and Poisson Distribution.  
It demonstrates how to prepare data, train the model, tune hyperparameters, make predictions, and visualize probabilities — all in a **single script**.  

---

## 📂 Project Structure  

- **`Predictive__Maintenance.py`**  
  - Loads the dataset `failure_dataset.csv`.  
  - Preprocesses features with `StandardScaler`.  
  - Splits data into train and test sets.  
  - Trains a `RandomForestRegressor`.  
  - Runs **hyperparameter tuning** with `GridSearchCV`.  
  - Predicts equipment failures from operational data.  
  - Computes and visualizes failure probabilities using **Poisson Distribution**.  

---

## ⚙️ Workflow  

1. **Load and Preprocess Data**  
   - Reads `failure_dataset.csv`.  
   - Applies scaling to ensure proper feature normalization.  

2. **Train and Tune the Model**  
   - Baseline `RandomForestRegressor`.  
   - Grid search for optimal hyperparameters.  

3. **Make Predictions**  
   - Predicts λ (failure rate) for new inputs.  
   - Generates probabilities of occurrences using the Poisson distribution.  

4. **Visualize Results**  
   - Creates probability distribution plots for predicted failures.  

---

## 💡 Business Benefits  

Applying predictive maintenance in this way helps:  

- **Reduce downtime** by forecasting equipment failures in advance.  
- **Save operational costs** through proactive maintenance planning.  
- **Improve reliability** of assets by continuously monitoring performance.  
- **Enable scalability** for deployment into real-time monitoring systems.  

---

## 📝 How to Run  

Install dependencies and execute the script:  

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python Predictive__Maintenance.py
