#%%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

os.chdir(r"/Users/raul/Documents/GitHub/lstm_inflacion")
data_path = r"./data/base_data.xlsx"
data = pd.read_excel(data_path, sheet_name='data').dropna()
#%%
# Seleccionar las columnas predictoras y la columna de inflación
predictor_columns = ['ISE', 'TRM_promedio', 'PRECIP_GAP']  # Cambia esto a tus nombres de columnas
target_column = 'Inflation'
data = data[predictor_columns + [target_column]]

# Escalar los datos entre 0 y 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
#%%
# Dividir los datos en entrenamiento y prueba
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :-1])
        Y.append(dataset[i + time_step, -1])
    return np.array(X), np.array(Y)

# Definir el paso de tiempo (time_step)
time_step = 12

# Crear los conjuntos de datos de entrenamiento y prueba
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Redimensionar la entrada a [muestras, pasos de tiempo, características]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], len(predictor_columns))
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], len(predictor_columns))
#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(4, input_shape=(time_step, len(predictor_columns))))
model.add(Dense(4))
model.add(Dense(1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=200)

#%%

# Hacer predicciones
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Desescalar las predicciones
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit(data[[target_column]])
train_predict = scaler_y.inverse_transform(train_predict)
test_predict = scaler_y.inverse_transform(test_predict)

# Desescalar los datos originales
y_train = scaler_y.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calcular el error de la raíz cuadrada media (RMSE)
rmse_train = np.sqrt(np.mean((train_predict - y_train) ** 2))
rmse_test = np.sqrt(np.mean((test_predict - y_test) ** 2))

print(f'RMSE del conjunto de entrenamiento: {rmse_train}')
print(f'RMSE del conjunto de prueba: {rmse_test}')
#%%
import matplotlib.pyplot as plt

# Graficar los resultados
train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict) + time_step, :] = train_predict

test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
test_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict

plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='Datos reales')
plt.plot(train_plot, label='Predicción de entrenamiento')
plt.plot(test_plot, label='Predicción de prueba')
plt.xlabel('Fecha')
plt.ylabel('Inflation')
plt.legend()
plt.show()

#%%
# Obtener los últimos 12 meses de datos para hacer la predicción
last_12_months = scaled_data[-time_step:, :-1]

# Crear una lista para almacenar las predicciones futuras
future_predictions = []

# Hacer predicciones para los próximos 12 meses
for i in range(12):
    # Redimensionar los datos de entrada
    last_12_months_reshaped = last_12_months.reshape((1, time_step, len(predictor_columns)))
    # Hacer la predicción
    next_prediction = model.predict(last_12_months_reshaped)
    # Almacenar la predicción
    future_predictions.append(next_prediction[0, 0])
    # Actualizar los datos de entrada para la próxima predicción
    last_12_months = np.append(last_12_months[1:], np.concatenate((last_12_months[-1, :-1], next_prediction), axis=None).reshape(1, -1), axis=0)

# Desescalar las predicciones futuras
future_predictions = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Crear un rango de fechas para las predicciones futuras
last_date = data.index[-1]
future_dates = pd.date_range(last_date, periods=12, freq='M')

# Crear un DataFrame para almacenar las predicciones futuras
future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Inflation_pred'])

# Mostrar las predicciones futuras
print(future_df)

# Graficar las predicciones futuras junto con los datos históricos
plt.figure(figsize=(12, 6))
plt.plot(data[target_column], label='Datos históricos')
plt.plot(future_df, label='Predicciones futuras', linestyle='--')
plt.xlabel('Fecha')
plt.ylabel('Inflation')
plt.legend()
plt.show()

# %%
