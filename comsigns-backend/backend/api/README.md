# Capa API (Interface Layer)

![FastAPI](https://img.shields.io/badge/FastAPI-Interface-009688?style=flat-square)

> **Responsabilidad**: Exponer la funcionalidad del backend al mundo exterior a través de una API RESTful, manejando la validación de peticiones, serialización de respuestas y códigos de estado HTTP.

Construida sobre **FastAPI**, esta capa es ligera y delega la lógica pesada a la capa de `services`.

## 📂 Estructura

- **`app.py`**: Punto de entrada de la aplicación. Configura CORS, middleware y monta los routers.
- **`routes/`**: Definición de endpoints agrupados por dominio.
  - `inference.py`: Endpoints para inferencia desde archivos `.pkl` y gestión de secuencias.
  - `video.py`: Endpoints para carga y procesamiento de video.

---

## 🔌 Endpoints Principales

### Inferencia (`/inference`)

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/batch` | **Batch Inference**: Procesa múltiples archivos `.pkl`. Retorna predicciones individuales y la secuencia semántica construida. |
| `GET` | `/sequence` | Obtiene el estado actual de la secuencia de palabras aceptadas. |
| `POST` | `/sequence/reset` | Reinicia la secuencia (nueva frase). |

### Video (`/api/video`)

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/infer` | **Video Inference**: Sube video -> Extrae Keypoints -> Infiere -> Decide. Pipeline completo end-to-end. |
| `POST` | `/info` | Obtiene metadatos técnicos (duración, FPS, resolución) de videos. |
| `GET` | `/config` | Obtiene la configuración actual de procesamiento de video (duración máx, formatos, etc.). |

### Sistema

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | **Health Check**: Retorna `200 OK` si el servicio está vivo y el modelo cargado. |
| `GET` | `/info` | Información detallada del modelo cargado (número de clases, dispositivo). |

---

## 🛠️ Modelos de Datos (Pydantic)

La API utiliza modelos Pydantic para validar entradas y salidas.

- **`PredictionResponse`**:
  - `gloss`: Palabra predicha.
  - `confidence`: Nivel de certeza (0-1).
  - `bucket`: Categoría de frecuencia (HEAD/MID/OTHER).
  - `accepted`: Booleano (aprobada por motor de decisión).

- **`VideoInferenceResponse`**:
  - Lista de `results` (predicciones por video).
  - Lista de `errors` (videos fallidos).

---

## ⚙️ Configuración del Servidor

El servidor utiliza `Uvicorn` como servidor ASGI.

- **Lazy Loading**: Los servicios pesados (Modelo, MediaPipe) se cargan en el primer request ("lazy") para acelerar el inicio del contenedor, excepto si se configura lo contrario.
- **CORS**: Configurado permisivamente para desarrollo (`*`), debe restringirse en producción.
