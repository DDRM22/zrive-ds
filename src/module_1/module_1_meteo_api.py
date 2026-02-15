
# Importamos librerias
import pandas as pd 
import requests 

# Definicion api y parámetros
API_URL = "https://archive-api.open-meteo.com/v1/archive"

COORDINATES = { 
               "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
               "London": {"latitude": 51.507351, "longitude": -0.127758}, 
               "Rio": {"latitude": -22.906847, "longitude": -43.172896}, 
               } 
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"] 

# Definimos una función que recorra las coordenadas y extraiga los datos

def get_data_meteo_api(city, start_date="2010-01-01", end_date="2020-12-31"):

    coords = COORDINATES[city]

    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(VARIABLES),
        "timezone": "auto"
    }

    response = requests.get(API_URL, params=params)

    # Comprobamos la respuesta de la API
    if response.status_code != 200:
        print(f"Error en la llamada a la API: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    #print(data)
    
    # Convertimos a DF
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = city
    df = df.dropna(subset=["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"])

    return df
