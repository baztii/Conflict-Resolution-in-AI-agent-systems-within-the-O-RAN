import pandas as pd

data = {
    "Traffic_Load": [10, 20, 30, 40, 50],  # Ejemplo de datos de carga de tráfico
    "Channel_Quality": [5, 3, 2, 4, 1],  # Ejemplo de datos de calidad del canal
    "MCS": [16, 24, 32, 64, 128],  # Ejemplo de esquemas de modulación y codificación
    "CPU_Time": [2, 4, 6, 8, 10],  # Ejemplo de tiempo de CPU en milisegundos
    "Airtime": [1, 2, 3, 4, 5],  # Ejemplo de tiempo en el aire en milisegundos
    "Computing_Platform": [1, 2, 1, 2, 1],  # Ejemplo de identificadores de plataforma de computación
    "Bandwidth": [20, 40, 60, 80, 100],  # Ejemplo de ancho de banda en MHz
    "Energy_Consumption": [100, 200, 300, 400, 500]  # Ejemplo de consumo de energía en mW
}

df = pd.DataFrame(data)
df.to_csv('energy_optimization_dataset.csv', index=False)

df = pd.read_csv("energy_optimization_dataset.csv")
print(df.head())
