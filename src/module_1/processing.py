
# Importamos librerias
import pandas as pd

# Reagrupamos por mes 

def agg_monthly(df):
    
    df = df.copy()
    
    # Verificamos las columnas
    print("\nColumnas disponibles:", df.columns.tolist())
    print("Primeras filas del df original:")
    print(df.head())
    
    # Verificamos que existe la variable "time"
    if "time" not in df.columns:
        raise ValueError(f"La columna 'time' no existe. Columnas disponibles: {df.columns.tolist()}")
    
    df["time"] = pd.to_datetime(df["time"])
    
    monthly = df.groupby(["city", pd.Grouper(key="time", freq="ME")]).agg({
        "temperature_2m_mean": "mean",      # media mensual
        "precipitation_sum": "sum",         # precipitación acumulada mensual
        "wind_speed_10m_max": "mean"        # media mensual
    }).reset_index()

    return monthly