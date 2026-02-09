# Suite de Pruebas API - ComSigns Backend

## Resumen de Endpoints Analizados

| Endpoint | Método | Entrada | Descripción |
|----------|--------|---------|-------------|
| `/infer` | POST | `.pkl` (multipart) | Inferencia individual |
| `/infer/batch/evaluate` | POST | Múltiples `.pkl` (multipart) | Batch con decision engine |
| `/api/video/infer` | POST | Video(s) (multipart) | Inferencia desde video |

---

## Suite de Pruebas (10 casos)

### TEST-001: Inferencia PKL exitosa (Happy Path)

| Campo | Valor |
|-------|-------|
| **Nombre** | `test_infer_single_pkl_success` |
| **Endpoint** | `POST /infer` |
| **Tipo** | Funcional - Happy Path |
| **Descripción Request** | |
| - Content-Type | `multipart/form-data` |
| - Body | Campo `file`: archivo `.pkl` válido con keypoints |
| - Query Param | `topk=5` (opcional, default) |
| **Validaciones** | |
| - HTTP Status | `200 OK` |
| - Campos obligatorios | `top1`, `topk`, `meta` |
| - `top1` | Contiene: `gloss` (string), `confidence` (float 0-1), `bucket` (string: HEAD\|MID\|OTHER), `is_other` (bool) |
| - `topk` | Lista de objetos con `rank`, `gloss`, `confidence`, `bucket` |
| - `meta.model` | String no vacío |
| - `meta.num_classes` | Integer > 0 |
| **Resultado esperado** | Predicción válida con estructura completa |

---

### TEST-002: Rechazo de archivo con extensión inválida

| Campo | Valor |
|-------|-------|
| **Nombre** | `test_infer_rejects_non_pkl_file` |
| **Endpoint** | `POST /infer` |
| **Tipo** | Negativa - Validación de entrada |
| **Descripción Request** | |
| - Content-Type | `multipart/form-data` |
| - Body | Campo `file`: archivo `sample.txt` o `sample.json` |
| **Validaciones** | |
| - HTTP Status | `400 Bad Request` |
| - Campo obligatorio | `detail` (string) |
| - Mensaje | Contiene "Invalid file type" y "Expected .pkl" |
| **Resultado esperado** | Rechazo con mensaje descriptivo |

---

### TEST-003: Rechazo de archivo PKL vacío

| Campo | Valor |
|-------|-------|
| **Nombre** | `test_infer_rejects_empty_file` |
| **Endpoint** | `POST /infer` |
| **Tipo** | Negativa - Validación de contenido |
| **Descripción Request** | |
| - Content-Type | `multipart/form-data` |
| - Body | Campo `file`: archivo `empty.pkl` con 0 bytes |
| **Validaciones** | |
| - HTTP Status | `400 Bad Request` |
| - Campo obligatorio | `detail` (string) |
| - Mensaje | Contiene "Empty file" |
| **Resultado esperado** | Rechazo explícito de archivo vacío |

---

### TEST-004: Batch inference con múltiples archivos PKL

| Campo | Valor |
|-------|-------|
| **Nombre** | `test_batch_evaluate_multiple_pkl_success` |
| **Endpoint** | `POST /infer/batch/evaluate` |
| **Tipo** | Funcional - Happy Path |
| **Descripción Request** | |
| - Content-Type | `multipart/form-data` |
| - Body | Campo `files`: 3 archivos `.pkl` válidos |
| - Query Param | `topk=5` |
| **Validaciones** | |
| - HTTP Status | `200 OK` |
| - Campos root | `results`, `errors`, `sequence`, `summary` |
| - `results` | Array de objetos con `file_name` (string), `prediction` (object) |
| - `prediction` | Contiene: `gloss`, `confidence`, `bucket`, `accepted` (bool), `reason` (string) |
| - `sequence.accepted` | Array de palabras aceptadas |
| - `sequence.length` | Integer >= 0 |
| - `summary.total` | 3 |
| - `summary.processed` | <= 3 |
| **Resultado esperado** | Procesamiento completo con secuencia construida |

---

### TEST-005: Batch inference con archivos mixtos (válidos + inválidos)

| Campo | Valor |
|-------|-------|
| **Nombre** | `test_batch_evaluate_partial_failure` |
| **Endpoint** | `POST /infer/batch/evaluate` |
| **Tipo** | Robustez - Degradación controlada |
| **Descripción Request** | |
| - Content-Type | `multipart/form-data` |
| - Body | 2 archivos `.pkl` válidos + 1 archivo `.txt` inválido |
| **Validaciones** | |
| - HTTP Status | `200 OK` (NO falla todo el batch) |
| - `results` | Array con 2 elementos (archivos válidos procesados) |
| - `errors` | Array con 1 elemento (archivo inválido) |
| - `errors[0].file_name` | Nombre del archivo `.txt` |
| - `errors[0].error` | Contiene "Invalid file type" |
| - `summary.processed` | 2 |
| - `summary.failed` | 1 |
| **Resultado esperado** | Fallos parciales no cancelan batch completo |

---

### TEST-006: Inferencia de video exitosa

| Campo | Valor |
|-------|-------|
| **Nombre** | `test_video_infer_mp4_success` |
| **Endpoint** | `POST /api/video/infer` |
| **Tipo** | Funcional - Happy Path |
| **Descripción Request** | |
| - Content-Type | `multipart/form-data` |
| - Body | Campo `files`: archivo `.mp4` válido (< 100MB, 0.1-30 seg) |
| - Query Param | `topk=5` |
| **Validaciones** | |
| - HTTP Status | `200 OK` |
| - Campos root | `results`, `errors` |
| - `results[0]` | Contiene: `video`, `class_id`, `class_name`, `gloss`, `score`, `accepted`, `reason` |
| - `video` | String = nombre archivo original |
| - `score` | Float entre 0 y 1 |
| - `accepted` | Boolean |
| **Resultado esperado** | Predicción exitosa desde video |

---

### TEST-007: Rechazo de video con extensión no soportada

| Campo | Valor |
|-------|-------|
| **Nombre** | `test_video_infer_rejects_invalid_extension` |
| **Endpoint** | `POST /api/video/infer` |
| **Tipo** | Negativa - Validación formato |
| **Descripción Request** | |
| - Content-Type | `multipart/form-data` |
| - Body | Campo `files`: archivo `video.gif` o `video.flv` |
| **Validaciones** | |
| - HTTP Status | `400 Bad Request` |
| - Campo obligatorio | `detail` (string) |
| - Mensaje | Contiene "Invalid file type" y lista de extensiones permitidas (.mp4, .mov, .avi, .webm, .mkv) |
| **Resultado esperado** | Rechazo con extensiones válidas informadas |

---

### TEST-008: Rechazo de video excesivamente grande

| Campo | Valor |
|-------|-------|
| **Nombre** | `test_video_infer_rejects_oversized_file` |
| **Endpoint** | `POST /api/video/infer` |
| **Tipo** | Negativa - Límite de recursos |
| **Descripción Request** | |
| - Content-Type | `multipart/form-data` |
| - Body | Campo `files`: archivo `.mp4` > 100MB |
| **Validaciones** | |
| - HTTP Status | `400 Bad Request` |
| - Campo obligatorio | `detail` (string) |
| - Mensaje | Contiene "File too large" y "max: 100MB" |
| **Resultado esperado** | Rechazo antes de procesamiento costoso |

---

### TEST-009: Validación de contrato - Health Check

| Campo | Valor |
|-------|-------|
| **Nombre** | `test_health_endpoint_contract` |
| **Endpoint** | `GET /health` |
| **Tipo** | Validación de contrato |
| **Descripción Request** | |
| - Headers | Ninguno requerido |
| - Body | Vacío |
| **Validaciones** | |
| - HTTP Status | `200 OK` |
| - Campos obligatorios | `status`, `model_loaded` |
| - `status` | String: "healthy" o "unhealthy" |
| - `model_loaded` | Boolean |
| - `num_classes` | Integer > 0 (cuando model_loaded=true) |
| - Tiempo respuesta | < 500ms |
| **Resultado esperado** | Respuesta estructurada de salud del servicio |

---

### TEST-010: Validación parámetro topk fuera de rango

| Campo | Valor |
|-------|-------|
| **Nombre** | `test_infer_rejects_invalid_topk_range` |
| **Endpoint** | `POST /infer` |
| **Tipo** | Validación de parámetros |
| **Descripción Request** | |
| - Content-Type | `multipart/form-data` |
| - Body | Campo `file`: archivo `.pkl` válido |
| - Query Param | `topk=50` (excede máximo de 20) |
| **Validaciones** | |
| - HTTP Status | `422 Unprocessable Entity` |
| - Campo | `detail` (array de errores de validación) |
| - Error | Indica que `topk` debe ser <= 20 |
| **Resultado esperado** | FastAPI valida constraints automáticamente |

---

## Matriz de Cobertura

| Categoría | Cantidad | IDs |
|-----------|----------|-----|
| Happy Path | 3 | TEST-001, TEST-004, TEST-006 |
| Validación negativa | 5 | TEST-002, TEST-003, TEST-007, TEST-008, TEST-010 |
| Robustez / Degradación | 1 | TEST-005 |
| Contrato API | 1 | TEST-009 |

---

## Archivos de Prueba Requeridos

| Archivo | Descripción | Uso |
|---------|-------------|-----|
| `valid_sample.pkl` | PKL con keypoints válidos (hand, body, face tensors) | TEST-001, TEST-004, TEST-005, TEST-010 |
| `empty.pkl` | Archivo de 0 bytes | TEST-003 |
| `sample.txt` | Archivo texto plano | TEST-002, TEST-005 |
| `valid_sign.mp4` | Video MP4 válido, 2-5 segundos, < 50MB | TEST-006 |
| `invalid.gif` | Archivo GIF | TEST-007 |
| `large_video.mp4` | Video > 100MB | TEST-008 |

---

## Notas de Implementación para Postman

1. **Colección**: Crear colección "ComSigns API Tests"
2. **Variables de entorno**: `{{base_url}}` = `http://localhost:8000`
3. **Pre-request scripts**: Ninguno requerido (sin autenticación)
4. **Tests automáticos**: Usar `pm.test()` para validar schemas
5. **Archivos**: Cargar en cada request vía form-data (modo binario)

**Ejemplo snippet Postman Tests (TEST-001)**:
```javascript
pm.test("Status 200", () => pm.response.to.have.status(200));
pm.test("Has top1", () => pm.expect(pm.response.json()).to.have.property("top1"));
pm.test("Has topk array", () => pm.expect(pm.response.json().topk).to.be.an("array"));
pm.test("Confidence is float", () => pm.expect(pm.response.json().top1.confidence).to.be.a("number"));
```</content>
<parameter name="filePath">/Users/marloveper__/Documents/proyectos/COMSIGNS-BACKEND/tests/api_test_suite.md