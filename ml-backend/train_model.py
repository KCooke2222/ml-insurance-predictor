"""
Trains a Neural Network and Random Forest to predict insurance charges.
Run from the repo root: python ml-backend/train_model.py
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import joblib

RESULTS_DIR = "ml-backend/model_results"
MODEL_DIR = "ml-backend/model"


def load_data():
    df = pd.read_csv("ml-backend/insurance.csv")
    df["charges"] = df["charges"].astype(int)

    label_encoder = LabelEncoder()
    for col in ["sex", "region", "smoker"]:
        df[col] = label_encoder.fit_transform(df[col])

    X = df.drop("charges", axis=1)
    y = df["charges"]
    return X, y


def preprocess(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    feature_means = X.mean()
    return X_train_s, X_test_s, y_train, y_test, scaler, feature_means


def train_nn(X_train, X_test, y_train, y_test):
    model = keras.Sequential([
        layers.Input(shape=[X_train.shape[1]]),
        layers.Dense(1024, activation="relu"),
        layers.Dense(1024, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mae")

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=200, batch_size=64,
        callbacks=[early_stop],
        validation_split=0.2,
        verbose=1,
    )

    y_pred = model.predict(X_test).flatten()
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }
    return model, history, metrics


def train_rf(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }
    return model, metrics


def permutation_importance(model, X_test, y_test, n_repeats=5):
    baseline = mean_absolute_error(y_test, model.predict(X_test))
    importances = []
    rng = np.random.default_rng(42)
    for i in range(X_test.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            scores.append(mean_absolute_error(y_test, model.predict(X_perm)))
        importances.append(np.mean(scores) - baseline)
    return np.array(importances)


def plot_loss_curve(history, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_title("Neural Network Training Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importances, feature_names, title, path):
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color="#2563EB")
    ax.set_xlabel("Mean Increase in MAE (permutation)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_artifacts(nn_model, rf_model, scaler, feature_means, feature_names,
                   nn_metrics, rf_metrics, X_test, y_test):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Models
    nn_model.save(f"{MODEL_DIR}/model_nn.keras")
    joblib.dump(rf_model, f"{MODEL_DIR}/model_rf.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(feature_means, f"{MODEL_DIR}/feature_means.pkl")

    # Metrics
    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump({"nn": nn_metrics, "rf": rf_metrics}, f, indent=2)

    # Scaler params (for JS)
    with open(f"{RESULTS_DIR}/scaler_params.json", "w") as f:
        json.dump({
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "feature_names": feature_names,
        }, f, indent=2)

    # Feature means (for JS)
    with open(f"{RESULTS_DIR}/feature_means.json", "w") as f:
        json.dump(feature_means.to_dict(), f, indent=2)

    # Feature importance charts
    nn_imp = permutation_importance(nn_model, X_test, y_test)
    plot_feature_importance(nn_imp, feature_names, "Neural Network — Feature Importance",
                            f"{RESULTS_DIR}/nn_feature_importance.png")

    rf_imp = permutation_importance(rf_model, X_test, y_test)
    plot_feature_importance(rf_imp, feature_names, "Random Forest — Feature Importance",
                            f"{RESULTS_DIR}/rf_feature_importance.png")

    print("All artifacts saved.")


def main():
    print("Loading data...")
    X, y = load_data()
    feature_names = list(X.columns)

    print("Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, feature_means = preprocess(X, y)

    print("Training Neural Network...")
    nn_model, history, nn_metrics = train_nn(X_train, X_test, y_train, y_test)
    print(f"  NN  — MAE: {nn_metrics['mae']}, RMSE: {nn_metrics['rmse']}, R²: {nn_metrics['r2']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_loss_curve(history, f"{RESULTS_DIR}/nn_loss_curve.png")

    print("Training Random Forest...")
    rf_model, rf_metrics = train_rf(X_train, X_test, y_train, y_test)
    print(f"  RF  — MAE: {rf_metrics['mae']}, RMSE: {rf_metrics['rmse']}, R²: {rf_metrics['r2']}")

    print("Saving all artifacts...")
    save_artifacts(nn_model, rf_model, scaler, feature_means, feature_names,
                   nn_metrics, rf_metrics, X_test, y_test)

    print("Done.")


if __name__ == "__main__":
    main()
