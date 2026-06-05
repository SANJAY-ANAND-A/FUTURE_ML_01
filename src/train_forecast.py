import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'train.csv')
data_path = os.path.abspath(data_path)
df = pd.read_csv(data_path, parse_dates=['date'])
if 'sales' in df.columns:
    df = df[['date', 'sales']]
else:
    df = df.iloc[:, :2]
df = df.groupby('date')['sales'].sum().reset_index()
df = df.sort_values('date')
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['lag_7'] = df['sales'].shift(7)
df['lag_14'] = df['sales'].shift(14)
df['rolling_7'] = df['sales'].shift(1).rolling(7).mean()
df = df.dropna()
forecast_horizon = 16
train = df.iloc[:-forecast_horizon]
test = df.iloc[-forecast_horizon:]
features = ['dayofweek', 'month', 'year', 'lag_7', 'lag_14', 'rolling_7']
X_train = train[features]
y_train = train['sales']
X_test = test[features]
y_test = test['sales']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs'), exist_ok=True)
model_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'rf_model.joblib')
joblib.dump(model, model_file)
plt.figure(figsize=(10,5))
plt.plot(test['date'], y_test.values, label='actual')
plt.plot(test['date'], pred, label='predicted')
plt.legend()
plot_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'forecast.png')
plt.savefig(plot_file)
print('mse', mse)
print('model saved to', model_file)
print('plot saved to', plot_file)