
# Importamos librerias

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from module_1_meteo_api import get_data_meteo_api
from processing import agg_monthly

#Función de representación
def data_plot(df):
    
    VARIABLES = [
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max"
    ]

    cities = df["city"].unique()
    
    fig, axes = plt.subplots(3, 1, figsize=(10,10), sharex = True)
    
    for i, var in enumerate(VARIABLES):
        for city in cities:
            city_data = df[df["city"] == city]
            axes[i].plot(city_data["time"], city_data[var], label=city)

        axes[i].set_title(var)
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    #plt.show()
    plt.savefig('Weather.png', dpi=300, bbox_inches='tight')
    print("\n Gráfico guardado como 'Weather.png'")
    plt.close()

def main():

    cities = ["Madrid", "London", "Rio"]
    all_data = []

    # Llamar a la API por cada ciudad
    for city in cities:
        print(f"Descargando datos de {city}...")
        df = get_data_meteo_api(city)
        all_data.append(df)

    # Unimos todos los datos
    combined = pd.concat(all_data)

    # Reducimos a mensual
    monthly = agg_monthly(combined)
    
    print("\nPrimeras filas:")
    print(monthly.head())
    print(f"\nCiudades: {monthly['city'].unique()}")
    print(f"Total de registros: {len(monthly)}")
    print(f"Fechas: desde {monthly['time'].min()} hasta {monthly['time'].max()}")
    
    # Graficamos
    data_plot(monthly)

if __name__ == "__main__":
    main()