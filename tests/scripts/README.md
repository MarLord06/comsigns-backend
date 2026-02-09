# Scripts de Automatización - ComSigns API Tests

Esta carpeta contiene scripts para automatizar la ejecución y validación de pruebas API.

## Scripts Disponibles

### `run_api_tests.py`
**Script principal para ejecutar la suite completa de pruebas API.**

```bash
# Ejecución básica
python run_api_tests.py

# Con output detallado
python run_api_tests.py --verbose

# Con reporte HTML
python run_api_tests.py --report

# Combinado
python run_api_tests.py --verbose --report
```

**Características:**
- ✅ Ejecuta todos los 10 tests de la suite
- ✅ Valida respuestas automáticamente
- ✅ Mide tiempos de respuesta
- ✅ Genera reportes HTML
- ✅ Exit codes apropiados para CI/CD
- ✅ Manejo robusto de errores

**Requisitos:**
- Backend corriendo en `http://localhost:8000`
- Archivos de prueba en `../test_files/`
- Python 3.8+ con `requests`

### `validate_responses.py`
**Validador de esquemas de respuesta API.**

```bash
# Validar respuesta de inferencia
python validate_responses.py /infer test_files/samples/inference_response_example.json

# Validar health check
python validate_responses.py /health health_response.json
```

**Características:**
- ✅ Valida contratos API
- ✅ Verifica tipos de datos
- ✅ Campos requeridos
- ✅ Valores permitidos
- ✅ Estructuras anidadas

**Endpoints soportados:**
- `/infer` - Inferencia individual
- `/infer/batch/evaluate` - Batch con evaluación
- `/api/video/infer` - Inferencia de video
- `/health` - Health check

## Configuración de Pytest

### `pytest.ini`
Archivo de configuración para ejecutar pruebas con pytest.

```bash
# Instalar pytest si no está disponible
pip install pytest pytest-html

# Ejecutar tests
pytest scripts/test_api_endpoints.py -v

# Con reporte HTML
pytest scripts/test_api_endpoints.py -v --html=reports/pytest_report.html
```

## Ejemplos de Uso

### Pipeline de CI/CD
```yaml
# En GitHub Actions, GitLab CI, etc.
- name: Run API Tests
  run: |
    cd tests
    python scripts/run_api_tests.py --report

- name: Upload Test Reports
  uses: actions/upload-artifact@v2
  with:
    name: test-reports
    path: tests/reports/
```

### Desarrollo Local
```bash
# Terminal 1: Iniciar backend
cd backend
python -m uvicorn api.app:app --reload

# Terminal 2: Ejecutar tests
cd tests
python scripts/run_api_tests.py --verbose --report

# Ver reporte
open reports/test_report.html
```

### Validación Manual
```bash
# Probar endpoint individual
curl -X POST http://localhost:8000/infer \
  -F "file=@test_files/samples/valid_sample.pkl" \
  -o response.json

# Validar respuesta
python scripts/validate_responses.py /infer response.json
```

## Manejo de Errores

### Tests Fallidos
- **Archivos faltantes:** Verificar que `test_files/` contenga los archivos requeridos
- **Backend no disponible:** Asegurar que el backend esté corriendo en el puerto correcto
- **Timeouts:** Aumentar timeout en el script si las respuestas son lentas
- **Errores de validación:** Revisar esquemas API y actualizar validaciones si cambian

### Debugging
```bash
# Ejecutar con verbose para más detalles
python run_api_tests.py --verbose

# Verificar conectividad
curl http://localhost:8000/health

# Probar endpoint manualmente
curl -X POST http://localhost:8000/infer \
  -F "file=@test_files/samples/valid_sample.pkl"
```

## Extensión de Scripts

### Agregar Nuevo Test
1. Agregar lógica en `run_api_tests.py`
2. Definir validaciones apropiadas
3. Actualizar documentación

### Agregar Nuevo Endpoint
1. Agregar esquema en `validate_responses.py`
2. Crear método de validación
3. Actualizar documentación

### Personalización
- Modificar `BASE_URL` para diferentes entornos
- Ajustar timeouts según necesidades
- Agregar autenticación si es requerida
- Personalizar reportes HTML

## Dependencias

```
requests>=2.25.0    # Para HTTP requests
pytest>=6.0.0       # Para tests framework (opcional)
pytest-html>=3.0.0  # Para reportes HTML (opcional)
```

Instalar con:
```bash
pip install requests pytest pytest-html
```</content>
<parameter name="filePath">/Users/marloveper__/Documents/proyectos/COMSIGNS-BACKEND/tests/scripts/README.md