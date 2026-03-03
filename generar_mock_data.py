import pandas as pd
import numpy as np
import os

# --- CONFIGURACIÓN ---
ARCHIVO_CATALOGO = os.path.join("data", "external", "catalogo_estaciones.csv")
CARPETA_RAW = os.path.join("data", "raw")

def generar_datos_completos_v2():
    print("☢️ AGREGANDO VARIABLES: CONTAMINACIÓN Y RADIACIÓN (1985-2026)...")
    
    if not os.path.exists(CARPETA_RAW):
        os.makedirs(CARPETA_RAW)

    # Leer catálogo
    df_cat = pd.read_csv(ARCHIVO_CATALOGO, dtype=str)
    df_cat['estado'] = df_cat['estado'].str.strip().str.upper()
    seleccion = df_cat.groupby('estado').head(3)
    
    # Fechas
    fechas = pd.date_range(start="01/01/1985", end="31/12/2026", freq="D")
    total_dias = len(fechas)
    dias = np.arange(total_dias)
    
    count = 0
    for index, row in seleccion.iterrows():
        id_estacion = str(row['id_estacion']).zfill(5)
        estado = str(row['estado']).replace(" ", "_")
        nombre_archivo = f"{estado}_{id_estacion}.txt"
        ruta = os.path.join(CARPETA_RAW, nombre_archivo)
        
        # 1. Simulación Clima Base (TMAX, TMIN, LLUVIA)
        estacionalidad = 12 * np.sin(2 * np.pi * dias / 365.25)
        tmax = 25 + estacionalidad + np.random.normal(0, 3, total_dias)
        
        # 2. NUEVA VARIABLE: RADIACIÓN SOLAR (W/m²)
        # Sigue a la temperatura pero baja drásticamente si llueve
        # Verano ~900 W/m2, Invierno ~500 W/m2
        rad_base = 600 + 300 * np.sin(2 * np.pi * dias / 365.25)
        rad_ruido = np.random.normal(0, 50, total_dias)
        radiacion = np.clip(rad_base + rad_ruido, 100, 1200) # Limitar entre 100 y 1200
        
        # 3. NUEVA VARIABLE: CONTAMINACIÓN (AQI - Índice de Calidad del Aire)
        # La contaminación sube en invierno (inversión térmica) y baja con lluvia y viento
        # Escala AQI: 0-50 (Bien), 50-100 (Regular), 100+ (Malo)
        aqi_base = 70 - 20 * np.sin(2 * np.pi * dias / 365.25) # Más alto en invierno
        aqi_ruido = np.random.exponential(15, total_dias) # Picos repentinos
        aqi = np.clip(aqi_base + aqi_ruido, 10, 300)
        
        # 4. LLUVIA (Afecta a la radiación y limpia la contaminación)
        prob_lluvia = 0.25 + 0.25 * np.sin(2 * np.pi * (dias - 30) / 365.25)
        es_lluvia = np.random.rand(total_dias) < prob_lluvia
        precip = np.where(es_lluvia, np.random.exponential(15, total_dias), 0)
        
        # Ajustes cruzados:
        # Si llueve, baja la radiación
        radiacion = np.where(precip > 5, radiacion * 0.4, radiacion)
        # Si llueve, baja la contaminación (la lluvia limpia el aire)
        aqi = np.where(precip > 5, aqi * 0.5, aqi)
        
        # Guardar archivo con las NUEVAS COLUMNAS
        df_temp = pd.DataFrame({
            'FECHA': fechas.strftime('%d/%m/%Y'),
            'PRECIP': np.round(precip, 1),
            'EVAP': np.round(np.random.uniform(2, 8, total_dias), 1),
            'TMAX': np.round(tmax, 1),
            'TMIN': np.round(tmax - 12, 1),
            'RAD_SOL': np.round(radiacion, 1),   # <--- NUEVA
            'AQI': np.round(aqi, 0)              # <--- NUEVA (Contaminación)
        })
        
        # Escribimos
        df_temp.to_csv(ruta, sep=' ', index=False)
        
        count += 1
        if count % 10 == 0: print(f"✅ Procesando {count} estaciones con Contaminación...")

    print("-" * 50)
    print(f"🏁 ¡LISTO! {count} archivos generados con las 3 VARIABLES.")

if __name__ == "__main__":
    generar_datos_completos_v2()