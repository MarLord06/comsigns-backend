# Archivos de Prueba - ComSigns API Tests

Esta carpeta contiene los archivos de prueba utilizados por la suite de pruebas API.

## Estructura

```
test_files/
├── samples/                 # Archivos PKL y de ejemplo para inferencia
│   ├── valid_sample.pkl    # Archivo PKL válido con keypoints
│   ├── empty.pkl           # Archivo vacío (0 bytes)
│   └── sample.txt          # Archivo con extensión inválida
└── videos/                  # Archivos de video para pruebas
    ├── valid_sign.mp4      # Video válido para inferencia
    ├── invalid.gif         # Video con extensión no soportada
    └── large_video.mp4     # Video que excede límite de tamaño
```

## Archivos Requeridos

### Para PKL Inference (`/infer`, `/infer/batch/evaluate`)

| Archivo | Descripción | Tests que lo usan |
|---------|-------------|-------------------|
| `valid_sample.pkl` | Archivo PKL válido con estructura de keypoints | TEST-001, TEST-004, TEST-005, TEST-010 |
| `empty.pkl` | Archivo de 0 bytes | TEST-003 |
| `sample.txt` | Archivo con extensión `.txt` | TEST-002, TEST-005 |

**Estructura esperada de `valid_sample.pkl`:**
```python
{
    "hand": torch.Tensor [T, hand_dim],  # Keypoints de mano
    "body": torch.Tensor [T, body_dim],  # Keypoints de cuerpo
    "face": torch.Tensor [T, face_dim],  # Keypoints de cara
    "lengths": torch.Tensor [1]          # Longitud de secuencia (opcional)
}
```

### Para Video Inference (`/api/video/infer`)

| Archivo | Descripción | Tests que lo usan |
|---------|-------------|-------------------|
| `valid_sign.mp4` | Video MP4 válido (2-5 seg, < 50MB) | TEST-006 |
| `invalid.gif` | Archivo con extensión no soportada | TEST-007 |
| `large_video.mp4` | Video que excede 100MB | TEST-008 |

**Requisitos para videos válidos:**
- **Formatos soportados:** .mp4, .mov, .avi, .webm, .mkv
- **Tamaño máximo:** 100 MB
- **Duración:** 0.1 - 30 segundos
- **Contenido:** Seña LSP-AEC segmentada (una seña por video)
- **Resolución recomendada:** 640x480 o similar
- **FPS recomendado:** 24-30 fps

## Generación de Archivos de Prueba

### Archivos PKL
Los archivos PKL se generan durante el preprocesamiento de video:

```bash
# Desde el directorio del proyecto
cd backend
python -c "
from services.video_preprocess import VideoPreprocessor
import pickle

# Procesar un video real para obtener keypoints
preprocessor = VideoPreprocessor()
features = preprocessor.process_video(video_bytes_o_path)

# Guardar como PKL
with open('valid_sample.pkl', 'wb') as f:
    pickle.dump(features, f)
"
```

### Videos de Prueba
Los videos de prueba deben contener señas LSP-AEC reales:

1. **Videos cortos válidos:** Grabar o extraer segmentos de 2-5 segundos con una seña clara
2. **Videos inválidos:** Cualquier archivo con extensión no soportada
3. **Videos grandes:** Archivos de video > 100MB (pueden ser videos largos o de alta resolución)

## Notas Importantes

- **Archivos simulados:** Los archivos actuales son marcadores de posición con documentación. Para pruebas reales, reemplazar con archivos válidos del tipo correcto.
- **Seguridad:** No subir archivos reales de video grandes al repositorio. Usar archivos de ejemplo pequeños o generarlos localmente.
- **Consistencia:** Asegurar que los archivos de prueba sean consistentes entre entornos de desarrollo y CI/CD.
- **Versionado:** Si cambian los formatos de archivo, actualizar tanto los archivos de prueba como las pruebas que los usan.

## Troubleshooting

### Error: "Invalid file type"
- Verificar que el archivo tenga la extensión correcta
- Para PKL: debe ser exactamente `.pkl`
- Para video: debe ser una de las extensiones soportadas

### Error: "Empty file"
- El archivo debe tener contenido real
- Archivos de 0 bytes son rechazados

### Error: "File too large"
- Videos deben ser < 100MB
- Comprimir o recortar videos si es necesario

### Error en procesamiento de PKL
- Verificar que el archivo contenga la estructura correcta de diccionario
- Los tensores deben tener las dimensiones esperadas
- Usar archivos generados por el propio sistema de preprocesamiento</content>
<parameter name="filePath">/Users/marloveper__/Documents/proyectos/COMSIGNS-BACKEND/tests/test_files/README.md