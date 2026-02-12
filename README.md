# ComSigns Backend

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00C853?style=flat-square&logo=google&logoColor=white)
![ComSigns](https://img.shields.io/badge/ComSigns-LSP--AEC-blue?style=flat-square)

> **API de inferencia para la traducción de Lengua de Señas Peruana (LSP-AEC) utilizando Deep Learning.**

## 📋 Descripción General

**ComSigns Backend** es el núcleo de inferencia del sistema ComSigns. Proporciona una API REST robusta diseñada para procesar video en tiempo real o archivos pre-procesados, extrayendo características corporales (keypoints) y clasificándolas en glosas de lengua de señas utilizando una arquitectura de red neuronal LSTM multimodal.

El sistema integra:
- **Extracción de características**: Uso de MediaPipe para detectar 21 puntos en cada mano, 33 de postura corporal y 468 faciales.
- **Modelo Deep Learning**: Arquitectura LSTM de tres ramas (Hand, Body, Face) con fusión multimodal.
- **Motor de Decisión**: Reglas deterministas para aceptar o rechazar predicciones basadas en confianza y heurísticas contextuales.
- **Resolución Semántica**: Mapeo inteligente de IDs numéricos a glosas legibles y categorías (HEAD, MID, OTHER).

---

## 🏗️ Arquitectura del Sistema

La arquitectura sigue un diseño en capas modular para separar responsabilidades:

```mermaid
graph TD
    Client[Cliente (Web/Mobile)] -->|HTTP POST| API[Capa API (FastAPI)]
    
    subgraph "ComSigns Backend"
        API -->|Request| Service[Capa de Servicios]
        
        subgraph "Services Layer"
            Service -->|Video| Preprocess[Video Preprocessor]
            Preprocess -->|Frames| Keypoints[Keypoint Extractor]
            Service -->|Features| InfService[Inference Service]
        end
        
        subgraph "Inference Layer"
            InfService -->|Tensors| Model[SignLanguageModel (LSTM)]
            Model -->|Logits| Predictor[Predictor]
        end
        
        subgraph "Semantic Layer"
            Predictor -->|Class ID| Resolver[Semantic Resolver]
            Resolver -->|Gloss| Semantics[Glosas & Mappings]
        end
        
        subgraph "Decision Layer"
            InfService -->|Prediction| Decision[Decision Engine]
            Decision -->|Rules| Sequence[Sequence Manager]
        end
    end
    
    Sequence -->|Response| API
```

## 📦 Módulos Principales

El backend está organizado en módulos especializados. Haz clic en cada uno para ver su documentación técnica detallada:

| Módulo | Descripción |
|--------|-------------|
| [**`api/`**](./comsigns-backend/backend/api/README.md) | **API Gateway**: Definición de endpoints FastAPI, rutas, modelos Pydantic y configuración del servidor. |
| [**`services/`**](./comsigns-backend/backend/services/README.md) | **Orquestación**: Lógica de negocio, procesamiento de video, servicios de inferencia y extracción de keypoints. |
| [**`inference/`**](./comsigns-backend/backend/inference/README.md) | **Deep Learning**: Arquitectura del modelo PyTorch, carga de checkpoints y ejecución de inferencia tensorial. |
| [**`semantic/`**](./comsigns-backend/backend/semantic/README.md) | **Semántica**: Resolución de predicciones numéricas a significados humanos, manejo de diccionarios y mapeos. |
| [**`decision_engine/`**](./comsigns-backend/backend/decision_engine/README.md) | **Reglas**: Motor de evaluación para aceptar/rechazar señas y gestión de la secuencia de frases. |

## 🚀 Instalación y Desarrollo Local

### Requisitos Previos
- Python 3.11+
- FFmpeg (para procesamiento de video)
- Git

### Pasos

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-org/comsigns-backend.git
   cd comsigns-backend
   ```

2. **Crear entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r comsigns-backend/requirements.txt
   ```
   > **Nota:** Para desarrollo en Mac con Apple Silicon, PyTorch usará aceleración MPS si está disponible, o CPU por defecto.

4. **Configurar variables de entorno:**
   Copia el ejemplo y ajusta según necesites:
   ```bash
   cp .env.example .env
   ```

5. **Ejecutar el servidor de desarrollo:**
   ```bash
   uvicorn comsigns-backend.backend.api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Verificar instalación:**
   Abre [http://localhost:8000/docs](http://localhost:8000/docs) para ver la documentación interactiva Swagger UI.

---

## ☁️ Deployment

El proyecto está configurado para despliegue automático en **Railway** usando `nixpacks`.

### Archivos de Configuración
- `railway.toml`: Configuración del servicio en Railway.
- `nixpacks.toml`: Definición del entorno de build (Python 3.11 + bibliotecas de sistema como FFmpeg).
- `Procfile`: Comando de inicio del proceso web.

### Variables de Entorno en Producción

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `PORT` | Puerto de escucha | `8000` (auto-asignado) |
| `COMSIGNS_DEVICE` | Dispositivo de cómputo | `cpu` |
| `LOG_LEVEL` | Verbose de logs | `INFO` |

---

## 📡 Resumen de API

### 🧠 Inferencia
- `POST /infer` - Inferencia simple desde archivo `.pkl`.
- `POST /infer/batch/evaluate` - Inferencia por lotes con evaluación de reglas y secuencia.
- `GET /sequence` - Obtener estado actual de la secuencia de palabras aceptadas.

### 📹 Video
- `POST /api/video/infer` - Sube un video, extrae keypoints y realiza inferencia end-to-end.
- `POST /api/video/info` - Obtiene metadatos técnicos de un archivo de video.

### ℹ️ Info & Health
- `GET /health` - Estado del servicio y carga del modelo.
- `GET /info` - Información detallada del modelo cargado y mapeo de clases.

---

## 🧪 Pruebas

El proyecto incluye una suite de pruebas completa.

```bash
# Ejecutar tests con pytest
cd tests
pytest
```

Para más detalles sobre la estrategia de pruebas, revisa la [Documentación de Pruebas](./tests/README.md).

---

## 📄 Licencia

Este proyecto es propiedad de **ComSigns Research Team**. Todos los derechos reservados.
