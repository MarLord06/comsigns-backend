# Módulo Semántico (Semantic Layer)

![JSON](https://img.shields.io/badge/Format-JSON-000000?style=flat-square&logo=json&logoColor=white)

> **Responsabilidad**: Interpretar la salida numérica ("cruda") del modelo y convertirla en conceptos humanos inteligibles (glosas), enriqueciendo la predicción con metadatos contextuales.

El módulo `semantic` actúa como el puente entre los números del modelo y el lenguaje natural.

## 🗂️ Componentes de Mapeo

El sistema utiliza dos artefactos JSON clave para la resolución:

1. **`class_mapping.json`**: Generado durante el entrenamiento.
   - Mapea `new_class_id` (0..N) a identificadores internos como `"HEAD_319"`, `"MID_22"`, o `"OTHER"`.
   - Contiene estadísticas de distribución (head/mid/tail).

2. **`dict.json`**: Diccionario maestro del dataset.
   - Mapea los identificadores internos (e.g., `319`) a la glosa textual real (e.g., `"YO"`).

---

## 🛠️ Componentes Principales

### 1. `SemanticMappingLoader`
Carga y valida los archivos de mapeo en memoria al iniciar la aplicación.

```python
loader = SemanticMappingLoader(class_mapping_path, dict_path)
loader.load()
print(loader.new_class_names[28]) # -> "HEAD_319"
print(loader.get_gloss(319))      # -> "YO"
```

### 2. `SemanticResolver`
El núcleo de este módulo. Recibe una predicción probabilística y devuelve un objeto semántico rico.

```mermaid
graph LR
    Input[Model Output: Class 28, Score 0.9] --> Resolver
    Resolver --> ClassMap[Lookup: Class 28 -> HEAD_319]
    ClassMap --> DictMap[Lookup: 319 -> "YO"]
    DictMap --> Output[SemanticPrediction]
```

**Lógica de Resolución:**
- Si la clase es **OTHER**: Se etiqueta como `is_other=True` y bucket `OTHER`.
- Si la clase es **HEAD/MID**: Se extrae el ID original y se busca su glosa.
- Si no hay glosa: Se usa el ID interno como fallback.

### 3. Tipos de Datos (`types.py`)

| Clase | Descripción |
|-------|-------------|
| `SemanticPrediction` | Resultado final: `gloss`, `confidence`, `bucket`, `is_other`. |
| `SemanticClassInfo` | Metadatos estáticos de una clase (cacheable). |
| `SemanticTopK` | Contenedor para una lista ordenada de predicciones semánticas. |

---

## 🏷️ Concepto de Buckets

El sistema clasifica las palabras en tres categorías según su frecuencia en el entrenamiento:

- **HEAD**: Palabras muy frecuentes (alta confianza, núcleo del vocabulario).
- **MID**: Palabras de frecuencia media.
- **OTHER**: Agrupación de todas las palabras poco frecuentes (TAIL) en una única clase "basura" para reducir falsos positivos.

*El módulo semántico expone esta información para que el `decision_engine` pueda aplicar reglas diferenciadas.*
