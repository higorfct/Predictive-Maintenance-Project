import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/failure_dataset.csv")

# Features and target
X = df[["operation_time", "temperature", "vibration"]]
y = df["failures"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error"
)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

# Save model and scaler
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved")
