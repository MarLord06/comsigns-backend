# ComSigns Backend - Suite de Pruebas API

## Descripción General

Esta carpeta contiene la suite completa de pruebas de API para el backend de ComSigns, enfocada en validar la lógica del sistema de interpretación de lenguaje de señas LSP-AEC de forma aislada.

## Estructura de la Suite

```
tests/
├── README.md                    # Este archivo
├── api_test_suite.md           # Especificación completa de pruebas
├── postman_collection.json     # Colección Postman con todos los tests
├── test_files/                 # Archivos de prueba de ejemplo
│   ├── samples/               # Archivos PKL de ejemplo
│   └── videos/                # Videos de prueba
├── scripts/                   # Scripts de automatización
│   ├── run_api_tests.py      # Script Python para ejecutar pruebas
│   └── validate_responses.py # Validación de esquemas
└── reports/                   # Reportes de ejecución (generados)
```

## Objetivos de la Suite

- **Validación funcional**: Verificar que todos los endpoints funcionen correctamente con entradas válidas
- **Validación negativa**: Asegurar que el sistema rechace entradas inválidas de manera controlada
- **Validación de contratos**: Confirmar que las respuestas cumplan con los esquemas definidos
- **Robustez**: Verificar comportamiento ante archivos corruptos o condiciones límite
- **Rendimiento**: Validar tiempos de respuesta aceptables

## Endpoints Cubiertos

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `POST /infer` | Inferencia individual desde archivo PKL | |
| `POST /infer/batch/evaluate` | Batch inference con decision engine | |
| `POST /api/video/infer` | Inferencia desde video | |
| `GET /health` | Health check del servicio | |

## Requisitos Previos

### Ambiente de Pruebas
- Backend corriendo en `http://localhost:8000`
- Modelo de IA cargado correctamente
- Archivos de configuración válidos

### Herramientas
- **Postman** (recomendado para pruebas manuales)
- **Python 3.8+** (para scripts de automatización)
- **pytest** (para pruebas automatizadas)

### Archivos de Prueba
Los archivos de prueba se encuentran en `test_files/`:
- `valid_sample.pkl`: Archivo PKL válido con keypoints
- `empty.pkl`: Archivo vacío
- `sample.txt`: Archivo con extensión inválida
- `valid_sign.mp4`: Video válido
- `invalid.gif`: Archivo con extensión no soportada
- `large_video.mp4`: Video que excede límites

## Ejecución de Pruebas

### Opción 1: Postman (Manual/Exploratorio)
1. Importar `postman_collection.json`
2. Configurar variable `base_url = http://localhost:8000`
3. Ejecutar colección completa o tests individuales

### Opción 2: Scripts Automatizados
```bash
cd tests/scripts
python run_api_tests.py
```

### Opción 3: pytest (Automatizado)
```bash
cd tests
pytest scripts/test_api_endpoints.py -v
```

## Matriz de Cobertura

| Categoría | Tests | Porcentaje |
|-----------|-------|------------|
| Happy Path | 3 | 30% |
| Validación Negativa | 5 | 50% |
| Robustez | 1 | 10% |
| Contrato API | 1 | 10% |
| **Total** | **10** | **100%** |

## Criterios de Aprobación

- ✅ **80%** de tests pasan (8/10)
- ✅ Todos los tests de Happy Path pasan
- ✅ Todos los tests de validación negativa pasan
- ✅ No hay regresiones en contratos API

## Reportes

Los reportes de ejecución se generan en `reports/`:
- `test_execution_report.html`: Reporte HTML detallado
- `test_results.json`: Resultados en formato JSON
- `coverage_report.xml`: Cobertura de código (si aplica)

## Mantenimiento

### Agregar Nuevos Tests
1. Documentar en `api_test_suite.md`
2. Agregar request en Postman
3. Crear script de validación si es necesario
4. Actualizar archivos de ejemplo

### Actualizar Tests Existentes
- Revisar cambios en contratos API
- Actualizar archivos de ejemplo si cambian formatos
- Mantener sincronización entre documentación y Postman

## Contacto

Para preguntas sobre la suite de pruebas:
- Consultar `api_test_suite.md` para especificaciones detalladas
- Revisar issues en el repositorio del proyecto</content>
<parameter name="filePath">/Users/marloveper__/Documents/proyectos/COMSIGNS-BACKEND/tests/README.md