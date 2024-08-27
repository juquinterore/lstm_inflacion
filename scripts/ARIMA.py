#%%
import os
import pandas as pd
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

os.chdir(r"/Users/raul/Documents/GitHub/lstm_inflacion")
data_path = r"./data/base_data.xlsx"
data = pd.read_excel(data_path, sheet_name='data').dropna()

# Seleccionar las variables
target_column = 'Inflation'
exog_columns = ['ISE', 'TRM_promedio', 'PRECIP_GAP']
data = data[exog_columns + [target_column]]
y = data[target_column]
exog = data[exog_columns]

# Dividir los datos en entrenamiento y prueba
train_size = int(len(y) * 0.8)
y_train, y_test = y[:train_size], y[train_size:]
exog_train, exog_test = exog[:train_size], exog[train_size:]
#%%
# auto_arima model para Inflation
model_arima = auto_arima(y_train, 
                         seasonal=True, 
                         trace=True, 
                         error_action='ignore', 
                         suppress_warnings=True, 
                         stepwise=True)

# Salida del modelo
print(model_arima.summary())

# Predicciones
predictions_arima = model_arima.predict(n_periods=len(y_test))

# Resultados
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Inflation')
plt.plot(y_test.index, predictions_arima, label='ARIMA Predictions')
plt.legend()
plt.title('ARIMA Model Predictions')
plt.show()

train_predictions = model_arima.predict_in_sample()

# Calculating RMSE for train set
train_rmse_arima = np.sqrt(mean_squared_error(y_train, train_predictions))
print(f'ARIMA Train RMSE: {train_rmse_arima}')

test_predictions = model_arima.predict(n_periods=len(y_test))

# Calculating RMSE for test set
test_rmse_arima = np.sqrt(mean_squared_error(y_test, test_predictions))
print(f'ARIMA Test RMSE: {test_rmse_arima}')

#%%
# auto_arimax model para Inflation con variables exógenas
model_arimax = auto_arima(y_train, 
                          exogenous=exog_train, 
                          seasonal=True, 
                          trace=True, 
                          error_action='ignore', 
                          suppress_warnings=True, 
                          stepwise=True)

# Salida del modelo
print(model_arimax.summary())

# Predicciones
predictions_arimax = model_arimax.predict(n_periods=len(y_test), exogenous=exog_test)

# Resultados
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Inflation')
plt.plot(y_test.index, predictions_arimax, label='ARIMAX Predictions')
plt.legend()
plt.title('ARIMAX Model Predictions')
plt.show()

train_predictions_arimax = model_arimax.predict_in_sample(exogenous=exog_train)

# Calculating RMSE for train set
train_rmse_arimax = np.sqrt(mean_squared_error(y_train, train_predictions_arimax))
print(f'ARIMAX Train RMSE: {train_rmse_arimax}')

test_predictions_arimax = model_arimax.predict(n_periods=len(y_test), exogenous=exog_test)

# Calculating RMSE for test set
test_rmse_arimax = np.sqrt(mean_squared_error(y_test, test_predictions_arimax))
print(f'ARIMAX Test RMSE: {test_rmse_arimax}')
# %%
