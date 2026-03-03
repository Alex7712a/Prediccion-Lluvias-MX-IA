import pandas as pd
import folium
import os

# Archivo de entrada
ARCHIVO = os.path.join("data", "processed", "dataset_maestro.csv")
CARPETA_MAPAS = "maps"
if not os.path.exists(CARPETA_MAPAS): os.makedirs(CARPETA_MAPAS)

# Coordenadas aproximadas de los estados (Capitales)
COORDENADAS = {
    'AGUASCALIENTES': [21.88, -102.29], 'BAJA_CALIFORNIA': [32.62, -115.45],
    'BAJA_CALIFORNIA_SUR': [24.14, -110.31], 'CAMPECHE': [19.83, -90.53],
    'CHIAPAS': [16.75, -93.12], 'CHIHUAHUA': [28.63, -106.08],
    'CIUDAD_DE_MÉXICO': [19.43, -99.13], 'COAHUILA_DE_ZARAGOZA': [25.42, -101.00],
    'COLIMA': [19.24, -103.72], 'DURANGO': [24.02, -104.65],
    'GUANAJUATO': [21.01, -101.25], 'GUERRERO': [17.55, -99.50],
    'HIDALGO': [20.10, -98.75], 'JALISCO': [20.65, -103.34],
    'MÉXICO': [19.28, -99.65], 'MICHOACÁN_DE_OCAMPO': [19.70, -101.19],
    'MORELOS': [18.92, -99.22], 'NAYARIT': [21.50, -104.89],
    'NUEVO_LEÓN': [25.68, -100.31], 'OAXACA': [17.07, -96.72],
    'PUEBLA': [19.04, -98.20], 'QUERÉTARO': [20.58, -100.38],
    'QUINTANA_ROO': [18.50, -88.30], 'SAN_LUIS_POTOSÍ': [22.15, -100.98],
    'SINALOA': [24.80, -107.43], 'SONORA': [29.07, -110.96],
    'TABASCO': [17.98, -92.94], 'TAMAULIPAS': [23.73, -99.13],
    'TLAXCALA': [19.31, -98.23], 'VERACRUZ_DE_IGNACIO_DE_LA_LLAVE': [19.54, -96.92],
    'YUCATÁN': [20.96, -89.59], 'ZACATECAS': [22.77, -102.57]
}

def generar_mapa():
    print("🗺️ GENERANDO MAPA INTERACTIVO DE MÉXICO...")
    
    df = pd.read_csv(ARCHIVO)
    
    # Calcular promedios por estado
    resumen = df.groupby('ESTADO')[['PRECIP', 'RAD_SOL', 'AQI']].mean().round(2)
    
    # Crear mapa base centrado en México
    m = folium.Map(location=[23.63, -102.55], zoom_start=5)
    
    contador = 0
    for estado, coords in COORDENADAS.items():
        if estado in resumen.index:
            datos = resumen.loc[estado]
            
            # Texto del popup
            html = f"""
            <h4>{estado}</h4>
            <b>Lluvia Prom:</b> {datos['PRECIP']} mm<br>
            <b>Radiación:</b> {datos['RAD_SOL']} W/m²<br>
            <b>Contaminación (AQI):</b> {datos['AQI']}
            """
            
            # Color del marcador según contaminación
            color = 'green' if datos['AQI'] < 50 else 'orange' if datos['AQI'] < 100 else 'red'
            
            folium.Marker(
                location=coords,
                popup=folium.Popup(html, max_width=200),
                tooltip=estado,
                icon=folium.Icon(color=color, icon='cloud')
            ).add_to(m)
            contador += 1

    ruta_mapa = os.path.join(CARPETA_MAPAS, "mapa_clima_contaminacion.html")
    m.save(ruta_mapa)
    
    print(f"✅ Mapa generado con {contador} estados.")
    print(f"📂 Archivo: {ruta_mapa}")
    
    # Abrir automáticamente
    try:
        os.startfile(ruta_mapa)
    except:
        pass

if __name__ == "__main__":
    generar_mapa()