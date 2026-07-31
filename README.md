# Predicción de Concentración de PM2.5 mediante Redes Neuronales Artificiales (MLP)

> **Proyecto académico** — Curso Redes Neuronales Artificiales
> Universidad del Bío-Bío

---

## 📋 Tabla de contenidos

- [Objetivo del proyecto](#objetivo-del-proyecto)
- [Dataset](#dataset)
- [Herramientas y tecnologías](#herramientas-y-tecnologías)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Metodología](#metodología)
- [Resultados](#resultados)
- [Función de predicción](#función-de-predicción)
- [Cómo usar este proyecto](#cómo-usar-este-proyecto)
- [Autor](#autor)

---

## Objetivo del proyecto

Desarrollar una solución de predicción de la concentración de **PM2.5** (material particulado fino, µg/m³) a partir de variables meteorológicas, utilizando una red neuronal **MLP (Multilayer Perceptron)** implementada con Keras/TensorFlow.

El proyecto cubre el flujo completo de un problema de ciencia de datos:

- Análisis Exploratorio de Datos (EDA)
- Preprocesamiento y transformación de variables
- Construcción y optimización de una red MLP
- Detección y manejo de overfitting
- Evaluación comparativa de modelos
- Interfaz de predicción individual reutilizable

---

## Dataset

**Fuente:** [Beijing PM2.5 Data Set — UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)

Datos horarios de calidad del aire y condiciones meteorológicas registrados en Beijing, China, entre los años 2010 y 2014.

| Característica | Detalle |
|---|---|
| Observaciones | 43.824 |
| Variables predictoras | 7 (6 numéricas + 1 categórica) |
| Variable objetivo | `pm2.5` (continua, µg/m³) |
| Tipo de problema | Regresión |

### Variables

| Variable | Descripción | Tipo |
|---|---|---|
| `DEWP` | Punto de rocío (°C) | Numérica |
| `TEMP` | Temperatura (°C) | Numérica |
| `PRES` | Presión atmosférica (hPa) | Numérica |
| `Iws` | Velocidad acumulada del viento (m/s) | Numérica |
| `Is` | Horas acumuladas de nieve | Numérica |
| `Ir` | Horas acumuladas de lluvia | Numérica |
| `cbwd` | Dirección combinada del viento (NE, NW, SE, cv) | Categórica |

---

## Herramientas y tecnologías

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-blue?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-purple?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-lightblue?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-green)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-teal)
![Keras Tuner](https://img.shields.io/badge/Keras_Tuner-Bayesian-yellow)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)

| Categoría | Herramientas |
|---|---|
| Lenguaje | Python 3.10 |
| Deep Learning | TensorFlow 2.x, Keras Sequential API |
| Búsqueda de hiperparámetros | Keras Tuner (Optimización Bayesiana) |
| Preprocesamiento | scikit-learn (`StandardScaler`, `train_test_split`) |
| Manipulación de datos | Pandas, NumPy |
| Visualización | Matplotlib, Seaborn |
| Serialización del modelo | `model.save()` (formato `.keras`), Joblib |
| Entorno | Jupyter Notebook |

---

## Estructura del repositorio

```
mlp-prediccion-pm2.5/
│
├── docs/
│   └── RedesNeuronales_Proyecto.pdf    # Presentación de resultados (Beamer)
│
├── scripts/
│   ├── main_script_pm25.ipynb          # Flujo completo: EDA, preprocesamiento, modelado, evaluación
│   ├── Funcion_pm25.py                 # Función de predicción individual reutilizable
│   ├── Tutorial_pm25.ipynb             # Guía de uso de la función prediccion()
│   ├── PRSA_data_2010.1.1-2014.12.31.csv  # Dataset original
│   │
│   └── modelo_guardado/
│       ├── modelo_pm25.keras           # Modelo MLP entrenado
│       ├── scaler_pm25.pkl             # StandardScaler ajustado en entrenamiento
│       └── columnas_pm25.pkl           # Orden de columnas post-encoding
│
└── README.md
```


---

## Metodología

### 1. Análisis Exploratorio de Datos (EDA)
- Revisión de valores faltantes (2.067 nulos en `pm2.5`, ≈4,7%), duplicados y tipos de variables
- Distribución del target: fuertemente asimétrica, con cola larga hacia valores extremos (máx. 994 µg/m³)
- Matriz de correlación: `Iws` presenta la correlación negativa más fuerte con PM2.5 (-0.25); se detectó alta colinealidad entre `TEMP` y `PRES` (-0.83)
- Análisis de PM2.5 por categoría de viento (`cbwd`): viento NW asociado a menores niveles de contaminación

### 2. Preprocesamiento
- Eliminación de filas con `pm2.5` nulo (variable objetivo, imputarla introduciría sesgo)
- Codificación de `cbwd` con *one-hot encoding* (`get_dummies`, `drop_first=True`)
- División: **70%** entrenamiento / **15%** validación / **15%** test
- Estandarización con `StandardScaler`, ajustado **solo en entrenamiento** (sin data leakage)
- Transformación del target: `log1p(pm2.5)` para reducir asimetría y estabilizar la varianza

### 3. Modelado MLP

**Modelo base:**
- 2 capas ocultas: 124 y 64 neuronas, activación ReLU
- Capa de salida: 1 neurona, activación lineal (regresión)
- Optimizador Adam (`lr=0.001`), función de pérdida MSE
- 100 épocas, `batch_size=32`

**Búsqueda de hiperparámetros:**
- Método: Optimización Bayesiana con Keras Tuner (30 combinaciones evaluadas)
- Espacio de búsqueda: N° de capas (1-4), neuronas iniciales (32/64/128/256), activación (ReLU/Tanh/ELU), learning rate (0.01/0.001/0.0001)
- **Mejor combinación encontrada:** 2 capas ocultas [256, 153 neuronas], activación ELU, `lr=0.001`, Early Stopping (`patience=4`, detenido en época 64)

### 4. Detección y manejo de overfitting
- Las curvas de pérdida train/validación muestran un gap leve y estable (~0.045 MSE en escala log) en ambos modelos
- El gap no se amplía con las épocas → overfitting leve y aceptable
- Se aplicó **Early Stopping** como medida preventiva; no se requirió Dropout ni regularización L2

---

## Resultados

Comparación de modelos en el conjunto de **test**:

| Modelo | RMSE (µg/m³) | MAE (µg/m³) | R² |
|---|---|---|---|
| Regresión Lineal (baseline) | 82,77 | 52,40 | 0,191 |
| **MLP Base** ✅ | **73,08** | 47,39 | **0,369** |
| MLP Optimizado | 74,62 | **47,12** | 0,343 |

**Modelo seleccionado:** MLP Base, por su mejor RMSE y R² global, y menor complejidad arquitectural.

**Interpretación:**
- Ambos MLP superan claramente a la regresión lineal, confirmando relaciones no lineales relevantes entre las variables meteorológicas y PM2.5
- Un R² de 0.369 es razonable dado que el dataset solo contiene variables meteorológicas, sin capturar factores de emisión (tráfico, industria, etc.) que también influyen en la contaminación real

---

## Función de predicción

`Funcion.py` expone la función `prediccion()`, que permite obtener una predicción individual a partir de variables meteorológicas crudas:

```python
from Funcion import prediccion

nueva_obs = {
    'DEWP': -10,
    'TEMP': -3,
    'PRES': 1025,
    'Iws': 35.5,
    'Is': 0,
    'Ir': 0,
    'cbwd': 'NW'
}

resultado = prediccion(nueva_obs)
# Predicción de PM2.5: 28.43 µg/m3 → Categoría: Bueno
```

La función se encarga automáticamente de:
- Codificar `cbwd` con *one-hot encoding* y alinear columnas
- Aplicar el mismo `StandardScaler` usado en el entrenamiento
- Revertir la transformación logarítmica (`expm1`)
- Clasificar el resultado según el Índice de Calidad del Aire (ICAP)

| Categoría | Rango PM2.5 (µg/m³) |
|---|---|
| 🟢 Bueno | 0 – 50 |
| 🟡 Regular (Alerta Temprana) | 51 – 79 |
| 🟠 Alerta | 80 – 109 |
| 🔴 Preemergencia | 110 – 169 |
| 🟣 Emergencia | Más de 170 |

Ver `Tutorial.ipynb` para ejemplos de uso con distintas condiciones meteorológicas.

---

## Cómo usar este proyecto

### Requisitos

```bash
pip install tensorflow keras-tuner scikit-learn pandas numpy matplotlib seaborn joblib
```

### Reproducir el flujo completo

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/pm25-mlp-prediccion.git
cd pm25-mlp-prediccion

# 2. Ejecutar el notebook principal (genera modelo, scaler y columnas guardadas)
jupyter notebook main_script_pm25.ipynb

# 3. Explorar el tutorial de uso de la función de predicción
jupyter notebook Tutorial.ipynb
```

> Asegúrate de que la carpeta `modelo_guardado/` exista en el mismo directorio antes de usar `Funcion.py` o `Tutorial.ipynb`. Si no existe, ejectura primero `main_script_pm25.ipynb` completo.

---

## Autor

**Tomás Saavedra Suazo**  
Estudiante de Estadística — Universidad del Bío-Bío  
