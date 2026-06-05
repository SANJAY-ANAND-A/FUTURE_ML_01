# FUTURE_ML_01

Sales forecasting using train, stores, transactions, holidays, and oil datasets.

## Run

1. Place `train.csv`, `stores.csv`, `transactions.csv`, `holidays_events.csv`, and `oil.csv` in the parent folder of this repository.
2. Create a virtual environment.
3. Install dependencies.
4. Run the training script.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src\train_forecast.py
```

## Deliverables

- `models/rf_store_forecast.joblib`
- `outputs/validation_forecast.csv`
- `outputs/store_1_forecast.png`
- `outputs/metrics.txt`
