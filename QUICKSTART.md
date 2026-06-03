# Quick Start

## 1. Clone the repo

```bash
git clone https://github.com/KCooke2222/ml-insurance-predictor.git
cd ml-insurance-predictor
```

## 2. Set up the backend

```bash
cd ml-backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run from the **repo root**:

```bash
python ml-backend/app.py
```

The API will be running at `http://localhost:5000`.

## 3. Run the frontend

In a second terminal:

```bash
cd insurance-predictor
npm install
npm run dev
```

Open http://localhost:5173. The frontend calls the Flask API for predictions.

## 4. Retrain the models (optional)

Only needed if you want to retrain from scratch or modify the training script. Run from the repo root with the venv active:

```bash
python ml-backend/train_model.py
```

This overwrites `ml-backend/model/` and `ml-backend/model_results/` with fresh artifacts.
