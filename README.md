# Insurance Cost Predictor

A machine learning app that estimates annual health insurance charges based on age, BMI, number of children, smoker status, sex, and region.

**Live demo:** [insurancepredictor.netlify.app](https://insurancepredictor.netlify.app)

## How it works

The models were trained on a synthetic US insurance dataset (1,338 records) commonly used for regression benchmarking. Enter age, BMI, number of children, and smoker status into the form and get an estimated annual charge.

A second page compares the neural network against a random forest trained on the same data, showing training curves, feature importance, and evaluation metrics side by side.

## Stack

| Layer    | Tech                             |
| -------- | -------------------------------- |
| ML       | TensorFlow / Keras, scikit-learn |
| Frontend | React, Vite, Tailwind CSS        |
| Deploy   | Netlify                          |

## Architecture

The training script (`ml-backend/train_model.py`) preprocesses the CSV, trains a neural network (two 1024-unit dense layers, early stopping) and a random forest (150 estimators), then saves each model alongside scaler parameters, permutation importance charts, and a metrics JSON.

## Features

- Live insurance cost predictor
- Side-by-side model comparison: neural network vs random forest
- Permutation feature importance charts for both models
- Training loss curve for the neural network
- Evaluation metrics: MAE, RMSE, R2

## Setup

See [QUICKSTART.md](QUICKSTART.md).
