# Clasificación de Texto con MLflow y API con FastAPI

Este proyecto se divide en dos partes complementarias:
1. Entrenamiento y seguimiento de un modelo de clasificación de texto usando MLflow.
2. Desarrollo de una API con FastAPI que expone funcionalidades de procesamiento de texto, incluyendo el uso de modelos de Hugging Face.

---

## 🧠 Parte 1: Clasificación de Texto con MLflow

### 🎯 Objetivo

- Entrenar un modelo supervisado que clasifique documentos en una de las 20 categorías del dataset 20 Newsgroups.
- Utilizar MLflow para registrar los parámetros, métricas y artefactos de cada experimento.

### 📚 Dataset
- **Nombre**: 20 Newsgroups
- **Origen**: `sklearn.datasets.fetch_20newsgroups`
- **Tamaño**: ~11.000 documentos

### 🔁 Flujo de trabajo
1. Carga y preprocesamiento del texto.
2. Vectorización con TF-IDF.
3. Entrenamiento con `MultinomialNB` usando distintos `alpha`.
4. Registro en MLflow.

### 📈 Métricas
- Accuracy entre `0.66` y `0.69`.
- Mejor resultado con `alpha = 0.1`.

### 📸 Capturas MLflow
- Runs registrados
- Detalle de un run
- Artefacto del modelo

---

## 🚀 Parte 2: API con FastAPI y Hugging Face

### 🎯 Objetivo

Desarrollar una API con al menos 5 endpoints, de los cuales 2 deben usar modelos de Hugging Face (`transformers.pipeline`).

### 📦 Dependencias
- `fastapi`
- `uvicorn`
- `transformers`
- `torch`

### 📂 Endpoints implementados

| Endpoint           | Descripción                                           | Tipo             |
|--------------------|--------------------------------------------------------|------------------|
| `/`                | Ping de salud                                         | Simple           |
| `/classify/`       | Clasificación zero-shot con etiquetas dinámicas       | Hugging Face     |
| `/translate/`      | Traducción inglés → francés                           | Hugging Face     |
| `/palindrome/`     | Verifica si una palabra es palíndroma                 | Lógica simple    |
| `/reversed/`       | Invierte un texto                                     | Lógica simple    |
| `/length/`         | Devuelve el número de caracteres de un texto          | Lógica simple    |

### 🖼️ Swagger UI (`/docs`)

Capturas requeridas:
- Vista general de los endpoints
- Ejecución de cada endpoint

### 🧪 Ejemplo de uso

```
http://localhost:8000/classify/?text=Me+gusta+el+vino&labels=positivo,negativo,neutral
```

---

## 🛠️ Requisitos de instalación

```bash
conda create -n mlflow-text python=3.10 -y
conda activate mlflow-text
pip install -r requirements.txt
```

---

## 🏁 Cómo ejecutar

### 1. Entrenamiento con MLflow
```bash
python main.py --nombre_job "ClasificacionTextoNB" --alpha_list 0.1 0.5 1.0
mlflow ui
```

👉 Ver en: [http://localhost:5000](http://localhost:5000)

### 2. Lanzar API FastAPI
```bash
uvicorn main:app --reload
```

👉 Ver Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ✍️ Autor
Eugenio Barquín  
Proyecto desarrollado como parte del módulo de prácticas con MLflow y FastAPI.
