# Descripción de la Aplicación y Entorno de Pruebas

## Informe de Calidad de Software — ComSigns Backend

---

## 1. Nombre y Objetivo del Proyecto

### **ComSigns Backend** — API de Inferencia para Lengua de Señas

**Objetivo**: ComSigns Backend es una API REST que proporciona servicios de inferencia para la traducción de lengua de señas peruana (LSP-AEC) a texto, utilizando modelos de aprendizaje profundo (PyTorch) y extracción de keypoints corporales (MediaPipe).

El sistema recibe archivos de video o keypoints pre-extraídos (.pkl), procesa las características a través de un modelo de redes neuronales entrenado, y retorna la predicción de la palabra correspondiente con su nivel de confianza.

**Alcance del sistema**:
- Procesamiento de video para extracción de keypoints
- Inferencia de modelo ML para clasificación de señas
- Motor de decisión con reglas de aceptación/rechazo
- Gestión de secuencias semánticas de palabras traducidas

---

## 2. Arquitectura del Sistema

### 2.1 Vista General

```
┌──────────────────────────────────────────────────────────────┐
│                    COMSIGNS BACKEND                           │
│              (API REST de Inferencia ML)                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     ┌─────────────────────────────────────┐ │
│  │   FastAPI   │────▶│         API Layer (routes/)         │ │
│  │  Endpoints  │     │  - inference.py (batch, sequence)   │ │
│  │   (app.py)  │     │  - video.py (video inference)       │ │
│  └─────────────┘     └──────────────┬──────────────────────┘ │
│                                     │                        │
│  ┌──────────────────────────────────▼──────────────────────┐ │
│  │               SERVICE LAYER (services/)                 │ │
│  │  - InferenceService: orquesta modelo + semántica        │ │
│  │  - BatchInferenceService: procesamiento por lotes       │ │
│  │  - VideoPreprocessor: frames → keypoints → tensores     │ │
│  │  - KeypointExtractor: MediaPipe wrapper                 │ │
│  └──────────────────────────────────┬──────────────────────┘ │
│                                     │                        │
│  ┌──────────────────────────────────▼──────────────────────┐ │
│  │              DOMAIN LAYER (decision_engine/)            │ │
│  │  - RuleEngine: reglas de aceptación (confidence, margin)│ │
│  │  - DecisionEvaluator: evalúa predicciones               │ │
│  │  - SequenceManager: estado de secuencia aceptada        │ │
│  └──────────────────────────────────┬──────────────────────┘ │
│                                     │                        │
│  ┌──────────────────────────────────▼──────────────────────┐ │
│  │           INFERENCE LAYER (inference/, semantic/)       │ │
│  │  - SignLanguageModel: arquitectura PyTorch              │ │
│  │  - Predictor: inferencia y top-k                        │ │
│  │  - SemanticResolver: class_id → gloss legible           │ │
│  └──────────────────────────────────┬──────────────────────┘ │
│                                     │                        │
│  ┌──────────────────────────────────▼──────────────────────┐ │
│  │              DATA LAYER (models/)                        │ │
│  │  - best.pt: checkpoint PyTorch (~50MB)                  │ │
│  │  - class_mapping.json: mapeo de clases                  │ │
│  │  - dict.json: diccionario gloss → significado           │ │
│  │  - MediaPipe task files: hand/face/pose landmarkers     │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Stack Tecnológico

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Python** | 3.11 | Lenguaje principal |
| **FastAPI** | 0.100+ | Framework API REST |
| **Uvicorn** | 0.23+ | Servidor ASGI |
| **Pydantic** | 2.0+ | Validación de datos y modelos de respuesta |
| **PyTorch** | 2.0+ | Inferencia de modelos de deep learning |
| **MediaPipe** | 0.10+ | Extracción de keypoints corporales |
| **OpenCV** | 4.8+ | Procesamiento de video |
| **NumPy** | 1.24+ | Operaciones numéricas |
| **FFmpeg** | - | Decodificación de video (sistema) |

### 2.3 Infraestructura de Despliegue

| Componente | Configuración |
|------------|---------------|
| **Plataforma** | Railway (PaaS) |
| **Build System** | Nixpacks |
| **Runtime** | Python 3.11 + FFmpeg |
| **Servidor** | Uvicorn (ASGI) |
| **Puerto** | Dinámico (asignado por Railway) |
| **Device** | CPU (inferencia sin GPU) |

---

## 3. Funcionalidades Principales a Probar

### 3.1 Endpoints del Sistema

#### **Endpoints de Sistema y Monitoreo**

| ID | Endpoint | Método | Descripción | Prioridad |
|----|----------|--------|-------------|-----------|
| E01 | `/` | GET | Información de la API y listado de endpoints | Baja |
| E02 | `/health` | GET | Health check del servidor y estado del modelo | Alta |
| E03 | `/info` | GET | Metadatos del modelo (clases, dispositivo) | Media |
| E04 | `/decision/config` | GET | Configuración del motor de decisión | Baja |

#### **Endpoints de Inferencia PKL**

| ID | Endpoint | Método | Descripción | Prioridad |
|----|----------|--------|-------------|-----------|
| E05 | `/infer` | POST | Inferencia individual sobre archivo .pkl | Alta |
| E06 | `/infer/evaluate` | POST | Inferencia + evaluación de reglas de aceptación | Alta |
| E07 | `/infer/batch` | POST | Inferencia batch sobre múltiples .pkl | Media |
| E08 | `/infer/batch/evaluate` | POST | Batch + evaluación + secuencia | Alta |

#### **Endpoints de Inferencia de Video**

| ID | Endpoint | Método | Descripción | Prioridad |
|----|----------|--------|-------------|-----------|
| E09 | `/api/video/infer` | POST | Inferencia sobre archivos de video | Alta |
| E10 | `/api/video/info` | POST | Obtener metadatos de video | Media |
| E11 | `/api/video/config` | GET | Configuración de procesamiento de video | Baja |

#### **Endpoints de Secuencia Semántica**

| ID | Endpoint | Método | Descripción | Prioridad |
|----|----------|--------|-------------|-----------|
| E12 | `/api/inference/batch` | POST | Batch con secuencia semántica | Alta |
| E13 | `/api/inference/sequence` | GET | Obtener secuencia actual | Media |
| E14 | `/api/inference/sequence/reset` | POST | Reiniciar secuencia | Media |
| E15 | `/sequence` | GET | Estado de secuencia (alternativo) | Baja |
| E16 | `/sequence/reset` | POST | Reset de secuencia (alternativo) | Baja |

### 3.2 Reglas de Negocio del Motor de Decisión

| Regla | Condición | Resultado |
|-------|-----------|-----------|
| R1 | `bucket == "OTHER"` | RECHAZAR (clase colapsada) |
| R2 | `confidence < 0.10` (HEAD/MID) | RECHAZAR (baja confianza) |
| R3 | `margin (top1 - top2) < 0.10` | RECHAZAR (predicción ambigua) |
| R4 | Ninguna de las anteriores | ACEPTAR |

### 3.3 Validaciones de Entrada

| Validación | Endpoint | Criterio | Código Error |
|------------|----------|----------|--------------|
| Extensión .pkl | `/infer`, `/infer/*` | Solo archivos .pkl | 400 |
| Archivo vacío | Todos POST con file | `len(contents) > 0` | 400 |
| Extensión video | `/api/video/infer` | .mp4, .mov, .avi, .webm, .mkv | 400 |
| Tamaño video | `/api/video/infer` | ≤ 100 MB | 400 |
| Duración video | `/api/video/infer` | 0.1s - 30s | 400 |
| Parámetro topk | Todos con `?topk=` | 1 ≤ topk ≤ 20 | 422 |

---

## 4. Entorno de Pruebas

### 4.1 Configuración del Entorno Local

| Componente | Configuración |
|------------|---------------|
| **Sistema Operativo** | macOS Sonoma 14.x / Windows 11 / Ubuntu 22.04 |
| **Python** | 3.11.x |
| **Gestor de paquetes** | pip 23.x |
| **Entorno virtual** | venv |

### 4.2 Configuración del Entorno de Producción (Railway)

| Componente | Configuración |
|------------|---------------|
| **Plataforma** | Railway |
| **Build** | Nixpacks (Python 3.11 + FFmpeg) |
| **Runtime** | Uvicorn ASGI |
| **URL Base** | `https://[app-name].railway.app` |
| **Health Check** | `/health` (timeout 300s) |

### 4.3 Herramientas de Prueba

| Herramienta | Versión | Propósito |
|-------------|---------|-----------|
| **Postman** | 10.x | Pruebas manuales de API |
| **Newman** | 6.x | Ejecución automatizada de colecciones |
| **Cypress** | 13.x | Pruebas E2E de API (cy.request) |
| **cURL** | - | Pruebas rápidas desde terminal |
| **pytest** | 8.x | Pruebas unitarias (si se implementan) |
| **httpie** | - | Cliente HTTP alternativo |

### 4.4 Datos de Prueba Requeridos

| Tipo de Archivo | Descripción | Uso |
|-----------------|-------------|-----|
| `sample_valid.pkl` | Keypoints extraídos de una seña válida | Happy path |
| `sample_empty.pkl` | Archivo vacío (0 bytes) | Test de validación |
| `test-video.mp4` | Video MP4 válido (2-5 segundos) | Inferencia de video |
| `invalid-file.txt` | Archivo de texto plano | Test de extensión |
| `large-video.mp4` | Video > 100MB | Test de límite de tamaño |
| `long-video.mp4` | Video > 30 segundos | Test de límite de duración |

#### Generación de Video de Prueba Válido

```bash
# Generar video sintético con FFmpeg
ffmpeg -f lavfi -i color=c=blue:s=320x240:d=3 -c:v libx264 -pix_fmt yuv420p test-video.mp4

# Verificar validez
ffprobe -v error test-video.mp4 && echo "✅ Video válido"
```

### 4.5 Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `PORT` | Puerto del servidor | 8000 |
| `COMSIGNS_DEVICE` | Dispositivo de inferencia | `cpu` |
| `COMSIGNS_CHECKPOINT` | Ruta al modelo | `models/.../best.pt` |
| `COMSIGNS_CLASS_MAPPING` | Ruta al mapeo de clases | `models/.../class_mapping.json` |
| `COMSIGNS_DICT` | Ruta al diccionario | `models/dict.json` |
| `LOG_LEVEL` | Nivel de logging | `INFO` |

---

## 5. Ejecución del Entorno de Pruebas

### 5.1 Levantar el Backend Localmente

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/comsigns-backend.git
cd comsigns-backend/comsigns-backend

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar servidor
uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5.2 Verificar que el Servidor está Operativo

```bash
# Health check
curl http://localhost:8000/health

# Respuesta esperada:
# {"status":"healthy","model_loaded":true,"num_classes":142}
```

### 5.3 Configurar Postman/Newman

```bash
# Exportar colección de Postman y ejecutar con Newman
newman run ComSigns_API_Tests.postman_collection.json \
  --environment Local.postman_environment.json \
  --reporters cli,html \
  --reporter-html-export results.html
```

---

## 6. Criterios de Aceptación

| Criterio | Descripción | Métrica |
|----------|-------------|---------|
| **Disponibilidad** | El endpoint `/health` responde correctamente | Status 200, `model_loaded: true` |
| **Funcionalidad** | Todos los endpoints responden según especificación | 100% de tests pasando |
| **Validación** | Errores de entrada retornan códigos apropiados | 400 para input inválido, 422 para tipos incorrectos |
| **Rendimiento** | Tiempo de respuesta aceptable | `/health` < 5s, `/infer` < 30s, `/api/video/infer` < 60s |
| **Estabilidad** | No hay errores 500 en flujos normales | 0 errores internos en happy path |

---

## 7. Limitaciones Conocidas

| Limitación | Descripción | Impacto en Pruebas |
|------------|-------------|-------------------|
| Sin autenticación | API pública sin JWT/API keys | No hay tests de auth |
| Inferencia en CPU | Modelo corre en CPU, más lento que GPU | Usar timeouts altos |
| Estado en memoria | Secuencia se pierde al reiniciar servidor | Considerar en tests de integración |
| Videos < 30s | Backend rechaza videos largos | Usar fixtures cortos |
| Primer request lento | Modelo se carga lazy en primer request | Health check antes de tests |

---

## 8. Estructura de Archivos del Proyecto

```
COMSIGNS-BACKEND/
├── railway.toml              # Configuración de Railway
├── nixpacks.toml             # Build config (Python + FFmpeg)
├── runtime.txt               # Python 3.11
├── start.sh                  # Script de inicio
├── Dockerfile                # Build alternativo
└── comsigns-backend/
    ├── requirements.txt      # Dependencias Python
    ├── backend/
    │   ├── api/
    │   │   ├── app.py        # FastAPI main app
    │   │   └── routes/
    │   │       ├── inference.py
    │   │       └── video.py
    │   ├── services/
    │   │   ├── inference_service.py
    │   │   ├── batch_service.py
    │   │   ├── video_preprocess.py
    │   │   └── keypoint_extractor.py
    │   ├── decision_engine/
    │   │   ├── evaluator.py
    │   │   ├── rules.py
    │   │   └── sequence.py
    │   ├── inference/
    │   │   ├── loader.py
    │   │   ├── model.py
    │   │   └── predictor.py
    │   └── semantic/
    │       ├── loader.py
    │       └── resolver.py
    └── models/
        ├── dict.json
        └── run_20260122_010532/
            ├── best_model.json
            ├── class_mapping.json
            └── checkpoints/
                └── best.pt
```

---

> **Documento preparado para**: Informe Final de Calidad de Software  
> **Componente**: Backend (API de Inferencia)  
> **Fecha de elaboración**: 8 de febrero de 2026  
> **Versión**: 1.0
