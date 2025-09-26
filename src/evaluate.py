import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/failure_dataset.csv")
X = df[["operation_time", "temperature", "vibration"]]
y = df["failures"]

# Split again to ensure same proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Scale test set
X_test_scaled = scaler.transform(X_test)

# Predictions
y_pred = model.predict(X_test_scaled)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Evaluation results - MAE: {mae:.2f}, R²: {r2:.2f}")
