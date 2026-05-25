import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import optuna

# -------------------------------
# PATH FIX (100% WORKING)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "House Price Prediction Dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Create model folder if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.joblib")

# -------------------------------
# DEBUG CHECK (IMPORTANT)
# -------------------------------
print("Dataset path:", DATA_PATH)
print("File exists:", os.path.exists(DATA_PATH))

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_PATH)

df = df.drop("Id", axis=1)

X = df.drop("Price", axis=1)
y = df["Price"]

# -------------------------------
# PREPROCESSING
# -------------------------------
ct = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Condition']),
        ('oe', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         ['Location', 'Garage'])
    ],
    remainder='passthrough'
)

X_transformed = ct.fit_transform(X)

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42
)

# -------------------------------
# SCALING
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# OPTUNA
# -------------------------------
def objective(trial):

    model_name = trial.suggest_categorical("regressor", ["XGB", "GBR"])

    if model_name == "GBR":
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("lr", 0.01, 0.2),
            random_state=42
        )
    else:
        model = XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("lr", 0.01, 0.2),
            verbosity=0,
            random_state=42
        )

    score = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error"
    ).mean()

    return -score


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best Params:", study.best_params)

# -------------------------------
# FINAL MODEL
# -------------------------------
best = study.best_params.copy()
model_type = best.pop("regressor")

if model_type == "GBR":
    model = GradientBoostingRegressor(**best, random_state=42)
else:
    model = XGBRegressor(**best, verbosity=0, random_state=42)

model.fit(X_train_scaled, y_train)

# -------------------------------
# EVALUATION
# -------------------------------
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE:", rmse)

# -------------------------------
# SAVE MODEL (FIXED)
# -------------------------------
joblib.dump({
    "model": model,
    "scaler": scaler,
    "column_transformer": ct
}, MODEL_PATH)

print("✅ Model saved successfully at:", MODEL_PATH)