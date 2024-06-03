import pandas as pd
import matplotlib.pyplot as plt

data_path = '.\\data\\base_data.xlsx'

data = pd.read_excel(data_path, sheet_name='data')


# Asegurarse de que las columnas estén en minúsculas para evitar problemas
data.columns = data.columns.str.lower()

# Convertir las columnas 'anio' y 'mes' a un índice de fecha
data['fecha'] = pd.to_datetime(data['anio'].astype(str) + '-' + data['mes'].astype(str) + '-01')
data.set_index('fecha', inplace=True)


## TRM ##
data['trm_cierre_var'] = data['trm_cierre'].pct_change(periods=12) * 100
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(data.index, data['trm_cierre'], color='tab:blue', label='TRM_cierre')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('TRM_cierre', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.plot(data.index, data['trm_cierre_var'], color='tab:orange', label='Variación anual de TRM_cierre')
ax2.set_ylabel('Variación mensual (%)', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.title('Evolución anual de TRM_cierre y su variación mensual')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
plt.show()

## IPC ##
data['ipc_var'] = data['ipc'].pct_change(periods=12) * 100
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(data.index, data['ipc'], color='tab:blue', label='IPC')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('IPC', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.plot(data.index, data['ipc_var'], color='tab:orange', label='Variación anual de IPC')
ax2.set_ylabel('Variación mensual (%)', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.title('Evolución anual de IPC y su variación mensual')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
plt.show()

## ISE ##
data['ise_var'] = data['ise'].pct_change(periods=12) * 100
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(data.index, data['ise'], color='tab:blue', label='ISE')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('ISE', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.plot(data.index, data['ise_var'], color='tab:orange', label='Variación anual de ISE')
ax2.set_ylabel('Variación mensual (%)', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.title('Evolución anual de ISE y su variación mensual')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
plt.show()

## Precipitacion ##
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(data.index, data['precip_gap'], color='tab:blue', label='precip_gap')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('ISE', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
plt.title('Evolución precip_gap')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
plt.show()

