# 🌧️ Evaluación Predictiva de Precipitaciones en México (1985-2026)

Este repositorio contiene el pipeline completo de Ingeniería de Datos y Machine Learning para predecir eventos de lluvia en México, analizando el impacto de variables climáticas y contaminantes atmosféricos.

## 🎯 Objetivos del Proyecto
* Predecir la ocurrencia de precipitación pluvial (>0 mm) a nivel nacional.
* Analizar la influencia de la **Radiación Solar** y la **Calidad del Aire (AQI/Contaminantes)** en la formación de precipitaciones.
* Evaluar modelos clásicos frente a redes neuronales para series temporales.

## 🗄️ Sobre el Dataset (Nota Técnica)
Debido a intermitencias técnicas en el acceso masivo a los servidores oficiales durante el desarrollo, se implementó un flujo de **Generación de Datos Sintéticos Controlados**. 
Los datos respetan las distribuciones estadísticas, estacionalidad y correlaciones físicas del clima real. El dataset procesado final consta de **1.4 millones de registros**.

👉 **[Haz clic aquí para descargar el Dataset Integrado (Excel) en Google Drive](https://docs.google.com/spreadsheets/d/1AApJLMhpNvJ5XSm1L7GF2Z4Xo4tl-6v3/edit?usp=drive_link&ouid=103611943744547339298&rtpof=true&sd=true)**

## ⚙️ Tecnologías y Modelos Implementados
* **Lenguaje:** Python (Pandas, Scikit-Learn, Folium, Seaborn)
* **Modelos Evaluados:**
  1. Regresión Logística (Baseline)
  2. Random Forest
  3. XGBoost (Gradient Boosting)
  4. **LSTM** (Redes Neuronales Recurrentes)

## 📊 Resultados Clave
* La **Radiación Solar** demostró ser el predictor de mayor importancia para los modelos.
* El modelo **LSTM** y el ensamble de **XGBoost** obtuvieron los F1-Scores más altos, demostrando una alta capacidad para capturar patrones temporales complejos y variaciones regionales, especialmente en las zonas Norte y Centro-Norte del país.
