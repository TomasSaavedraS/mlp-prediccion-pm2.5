import pandas as pd
import numpy as np
import joblib
from tensorflow import keras


def categoria_pm25(valor):
    """
    Clasifica un valor de PM2.5 (µg/m3) según el Índice de Calidad del Aire
    (ICAP) utilizado en este proyecto.
    """
    if valor <= 50:
        return "Bueno"
    elif valor <= 79:
        return "Regular (Alerta Temprana)"
    elif valor <= 109:
        return "Alerta"
    elif valor <= 169:
        return "Preemergencia"
    else:
        return "Emergencia"


def prediccion(observacion,
               modelo_path='modelo_guardado/modelo_pm25.keras',
               scaler_path='modelo_guardado/scaler_pm25.pkl',
               columnas_path='modelo_guardado/columnas_pm25.pkl'):
    """
    Recibe una nueva observación (dict o DataFrame) con las variables crudas
    (DEWP, TEMP, PRES, Iws, Is, Ir, cbwd) y devuelve la predicción de PM2.5
    en su escala original (µg/m3).
    """

    # Convertir a DataFrame si llega como dict
    if isinstance(observacion, dict):
        observacion = pd.DataFrame([observacion])

    # Codificar la variable categórica cbwd igual que en el entrenamiento
    observacion = pd.get_dummies(observacion, columns=['cbwd'], dtype=int)

    # Cargar columnas finales usadas en el entrenamiento (incluye dummies de cbwd)
    columnas = joblib.load(columnas_path)

    # Alinear columnas: agrega dummies faltantes con 0 y respeta el orden de entrenamiento
    observacion = observacion.reindex(columns=columnas, fill_value=0)

    # Cargar scaler y transformar
    scaler = joblib.load(scaler_path)
    obs_scaled = scaler.transform(observacion)

    # Cargar modelo y predecir (el modelo fue entrenado con el target en escala log1p)
    modelo = keras.models.load_model(modelo_path)
    pred_log = modelo.predict(obs_scaled, verbose=0)[0][0]

    # Revertir la transformación logarítmica para obtener el valor real de PM2.5
    pred = np.expm1(pred_log)
    categoria = categoria_pm25(pred)

    print(f"Predicción de PM2.5: {pred:.2f} µg/m3 → Categoría: {categoria}")

    return pred
