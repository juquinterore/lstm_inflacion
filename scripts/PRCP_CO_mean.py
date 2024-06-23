#%%
import pandas as pd
import os

#%%
required_csv = pd.read_csv(r"/Users/raul/Downloads/ghcn-m_v4_prcp_inventory.csv", delimiter=";")
csv_files_vector = required_csv["name"]
csv_directory = r"/Users/raul/Downloads/ghcn-m_v4.00.00_prcp_s16970101_e20240531_c20240606"
columns_to_extract = [1, 5, 6]

# DataFrame vacío para almacenar todos los datos
all_data = pd.DataFrame()

# Iterar sobre el vector de nombres de archivos CSV
for filename in csv_files_vector:
    file_path = os.path.join(csv_directory, filename)
    
    # Verificar si el archivo existe en el directorio
    if os.path.isfile(file_path):
        # Leer el archivo CSV, extrayendo solo las columnas necesarias
        df = pd.read_csv(file_path, usecols=columns_to_extract)
        
        # Nombrar las columnas extraídas para consistencia
        df.columns = ['Location', 'YYYYMM', 'PRCP']
        
        # Agregar los datos al DataFrame acumulativo
        all_data = pd.concat([all_data, df], ignore_index=True)
    else:
        print(f"El archivo {filename} no se encontró en el directorio.")

print(all_data)
#%%
data_mean = all_data.groupby(["YYYYMM"])[["PRCP"]].mean()
print(data_mean)
#%%
data_mean.to_csv(r"/Users/raul/Downloads/PCRP_CO_mean.csv", sep=";")
#%%