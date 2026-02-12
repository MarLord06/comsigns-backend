# Capa de Servicios (Services Layer)

![Python](https://img.shields.io/badge/Python-Logic-green?style=flat-square)

> **Responsabilidad**: Orquestar la lógica de negocio, coordinando la pre-procesamiento de datos (video/keypoints), la inferencia del modelo y la integración con el motor de decisiones.

La capa `services` es el "cerebro operativo" de la aplicación. Conecta los endpoints de la API con los módulos de bajo nivel (`inference`, `semantic`, `decision_engine`).

## 🧱 Componentes Principales

### 1. `InferenceService`
El servicio central de inferencia (Singleton).
- **Rol**: Facade para el subsistema de inferencia.
- **Flujo**: Carga el modelo y resolutor semántico -> Recibe tensores -> Ejecuta predicción -> Retorna respuesta semántica.

```python
service = get_inference_service()
response = service.infer_from_bytes(pkl_bytes, topk=5)
# output: InferenceResponse(top1=Prediction(...), topk=[...])
```

### 2. `VideoPreprocessor` & `KeypointExtractor`
Encargados de transformar video crudo en tensores procesables por el modelo.

**Pipeline de Video:**
1. **Decodificación**: Lee el video usando `OpenCV`.
2. **Sampling**: Muestrea frames a una tasa objetivo (ej. 10 FPS).
3. **Extracción**: Para cada frame, `KeypointExtractor` usa **MediaPipe** para detectar:
   - Manos (Hands)
   - Pose (Body)
   - Rostro (Face/Mesh)
4. **Normalización**: Convierte los puntos a tensores PyTorch y aplica padding/trimming (`max_frames=150`).

### 3. `BatchInferenceService`
Maneja la inferencia masiva y el estado de la secuencia.
- Procesa múltiples archivos `.pkl` en orden.
- Para cada uno:
  1. Llama a `InferenceService`.
  2. Pasa el resultado al `DecisionEvaluator`.
  3. Actualiza el `SequenceManager`.
- Retorna resultados individuales y la secuencia acumulada.

### 4. `PredictionService`
Clase base para servicios que requieren resolución semántica. Provee métodos utilitarios para formatear predicciones raw en objetos de respuesta de la API.

---

## 🔄 Flujo de Trabajo Típico

Cuando llega un request a `/api/video/infer`:

1. **API** recibe el archivo de video.
2. Llama a `VideoPreprocessor.process_video(Video)`.
   - -> `KeypointExtractor` (MediaPipe).
   - -> Retorna tensores `{hand, body, face}`.
3. API llama a `InferenceService.infer(Tensores)`.
   - -> `Predictor` (Modelo PyTorch).
   - -> `SemanticResolver` (Mapeo a glosa).
   - -> Retorna `InferenceResponse`.
4. API llama a `DecisionEvaluator.process(...)`.
   - -> `RuleEngine` (Aceptar/Rechazar).
   - -> `SequenceManager` (Actualizar historia).
5. API retorna respuesta JSON final.
