import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Librerías de Machine Learning (Sklearn & XGBoost)
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score

# Configuración
ARCHIVO_INPUT = os.path.join("data", "processed", "dataset_maestro.csv")
CARPETA_SALIDA = "resultados_rubrica"
if not os.path.exists(CARPETA_SALIDA): os.makedirs(CARPETA_SALIDA)

def ejecutar_analisis_completo():
    print("🔬 INICIANDO ANÁLISIS CIENTÍFICO (SEGÚN RÚBRICA)...")
    
    # 1. CARGA E INTERPOLACIÓN (Manejo de Gaps)
    print("1️⃣ Preprocesamiento y Feature Engineering...")
    df = pd.read_csv(ARCHIVO_INPUT)
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    
    # Requisito: "Manejen gaps con interpolación"
    # (Aunque nuestros datos están completos, ejecutamos el comando para cumplir)
    df = df.sort_values(['ESTADO', 'FECHA'])
    df = df.interpolate(method='linear') # Rellena huecos numéricos
    df = df.fillna(method='bfill')       # Rellena bordes

    # 2. CREACIÓN DE FEATURES (Lo que pide el profe)
    # Requisito: 'radiacion_lag1' y 'pm_media_estado'
    print("   - Generando lags y promedios regionales...")
    df['RAD_LAG1'] = df.groupby('ESTADO')['RAD_SOL'].shift(1)
    df['AQI_LAG1'] = df.groupby('ESTADO')['AQI'].shift(1) # Contaminación ayer
    
    # Promedio histórico de contaminación por estado (para ver si es zona sucia)
    df['pm_media_estado'] = df.groupby('ESTADO')['AQI'].transform('mean')
    
    # Target: Lluvia Binarizada (>0)
    df['LLOVERA'] = (df['PRECIP'] > 0).astype(int)
    df = df.dropna() # Eliminar la primera fila que quedó vacía por el lag

    # 3. SPLIT TEMPORAL (Train <= 2019, Test >= 2020)
    print("2️⃣ División Temporal (Train 1985-2019 | Test 2020+)...")
    train = df[df['FECHA'].dt.year <= 2019]
    test = df[df['FECHA'].dt.year >= 2020]
    
    features = ['RAD_SOL', 'AQI', 'TMAX', 'TMIN', 'MES', 'RAD_LAG1', 'pm_media_estado']
    X_train = train[features]
    y_train = train['LLOVERA']
    X_test = test[features]
    y_test = test['LLOVERA']

    # Escalado de datos (Obligatorio para Redes Neuronales LSTM)
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Diccionario de resultados
    resultados = []

    # --- MODELOS CLÁSICOS ---
    modelos = {
        "Regresion_Logistica": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=50, max_depth=10),
        "XGBoost": XGBClassifier(n_estimators=50, eval_metric='logloss')
    }

    for nombre, modelo in modelos.items():
        print(f"   🚀 Entrenando {nombre}...")
        modelo.fit(X_train_s, y_train)
        preds = modelo.predict(X_test_s)
        f1 = f1_score(y_test, preds)
        resultados.append({'Modelo': nombre, 'F1_Score': f1})
        
        # Matriz de Confusión (Heatmap)
        plt.figure(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Greens')
        plt.title(f'Matriz Confusión - {nombre}')
        plt.savefig(os.path.join(CARPETA_SALIDA, f"CM_{nombre}.png"))
        plt.close()

    # --- MODELO 4: LSTM (Deep Learning) ---
    print("   🧠 Entrenando Red Neuronal LSTM (Requisito Avanzado)...")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense

        # Redimensionar para LSTM [muestras, pasos_tiempo, caracteristicas]
        X_train_lstm = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
        X_test_lstm = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))

        model = Sequential()
        model.add(LSTM(50, input_shape=(1, X_train_s.shape[1]), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Entrenar (Epocas bajas para demo rápida)
        model.fit(X_train_lstm, y_train, epochs=3, batch_size=64, verbose=0)
        
        # Predecir
        preds_prob = model.predict(X_test_lstm)
        preds_lstm = (preds_prob > 0.5).astype(int)
        
        f1_lstm = f1_score(y_test, preds_lstm)
        print(f"      ✅ LSTM Terminado. F1-Score: {f1_lstm:.4f}")
        resultados.append({'Modelo': 'LSTM_DeepLearning', 'F1_Score': f1_lstm})

    except ImportError:
        print("      ⚠️ AVISO: No tienes TensorFlow instalado. Saltando LSTM.")
        print("      (Para instalarlo usa: pip install tensorflow)")
    except Exception as e:
        print(f"      ❌ Error en LSTM: {e}")

    # 4. VISUALES FINALES
    print("3️⃣ Generando Gráficas Comparativas...")
    df_res = pd.DataFrame(resultados)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x='Modelo', y='F1_Score', palette='magma')
    plt.title('Comparativa de Modelos (Incluyendo LSTM)')
    plt.ylim(0, 1)
    for i, v in enumerate(df_res['F1_Score']):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    
    plt.savefig(os.path.join(CARPETA_SALIDA, "COMPARATIVA_FINAL_F1.png"))
    print(f"✅ ¡ANÁLISIS COMPLETADO! Revisa la carpeta '{CARPETA_SALIDA}'")
    
    # Abrir carpeta
    try:
        os.startfile(CARPETA_SALIDA)
    except:
        pass

if __name__ == "__main__":
    ejecutar_analisis_completo()