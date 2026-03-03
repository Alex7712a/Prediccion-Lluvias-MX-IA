import pandas as pd
import os
import glob

# --- CONFIGURACIÓN ---
CARPETA_RAW = os.path.join("data", "raw")
CARPETA_PROCESSED = os.path.join("data", "processed")
ARCHIVO_SALIDA = os.path.join(CARPETA_PROCESSED, "dataset_maestro.csv")

def fusionar_todo():
    print("🏭 INICIANDO FUSIÓN MASIVA DE DATOS (ETL)...")
    
    if not os.path.exists(CARPETA_PROCESSED):
        os.makedirs(CARPETA_PROCESSED)
        
    # Buscar todos los archivos generados
    archivos = glob.glob(os.path.join(CARPETA_RAW, "*.txt"))
    print(f"📂 Se encontraron {len(archivos)} archivos raw para procesar.")
    
    lista_dfs = []
    
    for ruta in archivos:
        try:
            # Extraer Estado e ID del nombre del archivo
            nombre = os.path.basename(ruta).replace(".txt", "")
            if "_" in nombre:
                estado, id_est = nombre.rsplit("_", 1)
            else:
                estado, id_est = "DESCONOCIDO", nombre

            # Leer el archivo (Separado por espacios)
            df = pd.read_csv(ruta, sep=' ')
            
            # Formatear Fecha
            df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%Y')
            
            # Agregar Metadatos (Columnas de Fusión)
            df['ESTADO'] = estado
            df['ID_ESTACION'] = id_est
            
            # Feature Engineering Básico (Extraer Mes y Año)
            df['MES'] = df['FECHA'].dt.month
            df['AÑO'] = df['FECHA'].dt.year
            
            # Asegurar que tenemos las columnas del proyecto
            cols_ordenadas = [
                'FECHA', 'ESTADO', 'ID_ESTACION', 
                'PRECIP', 'TMAX', 'TMIN', 'RAD_SOL', 'AQI', 
                'MES', 'AÑO'
            ]
            
            # Solo agregar si el archivo tiene las columnas correctas
            if all(col in df.columns for col in ['RAD_SOL', 'AQI']):
                lista_dfs.append(df[cols_ordenadas])
            
        except Exception as e:
            print(f"⚠️ Error en {nombre}: {e}")

    # LA GRAN FUSIÓN (CONCATENACIÓN)
    if lista_dfs:
        print("🔄 Fusionando 1.4 millones de registros...")
        df_final = pd.concat(lista_dfs, ignore_index=True)
        
        # Ordenar cronológicamente
        df_final = df_final.sort_values(by=['ESTADO', 'FECHA'])
        
        # Guardar CSV Maestro
        df_final.to_csv(ARCHIVO_SALIDA, index=False)
        print("-" * 50)
        print(f"✅ ¡FUSIÓN COMPLETADA! Dataset creado: {ARCHIVO_SALIDA}")
        print(f"📊 Dimensiones: {df_final.shape[0]:,} filas x {df_final.shape[1]} columnas")
        print("   Variables: Lluvia, Temperatura, Radiación, Contaminación.")
    else:
        print("❌ No se encontraron datos válidos.")

if __name__ == "__main__":
    fusionar_todo()