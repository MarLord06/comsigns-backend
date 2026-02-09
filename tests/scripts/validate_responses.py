#!/usr/bin/env python3
"""
ComSigns API Response Validator

Script para validar esquemas de respuesta de la API de ComSigns.
Útil para verificar contratos API y validar respuestas.

Uso:
    python validate_responses.py <endpoint> <response_file.json>

Ejemplos:
    python validate_responses.py /infer response_sample.json
    python validate_responses.py /health health_response.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

class ResponseValidator:
    """Validador de esquemas de respuesta API"""

    def __init__(self):
        self.schemas = {
            "/infer": self._validate_inference_schema,
            "/infer/batch/evaluate": self._validate_batch_schema,
            "/api/video/infer": self._validate_video_schema,
            "/health": self._validate_health_schema,
        }

    def validate_response(self, endpoint: str, response_data: Dict[str, Any]) -> List[str]:
        """Valida una respuesta contra el esquema esperado"""
        if endpoint not in self.schemas:
            return [f"Endpoint no reconocido: {endpoint}"]

        try:
            return self.schemas[endpoint](response_data)
        except Exception as e:
            return [f"Error de validación: {str(e)}"]

    def _validate_inference_schema(self, data: Dict[str, Any]) -> List[str]:
        """Valida esquema de respuesta de /infer"""
        errors = []

        # Campos requeridos de nivel superior
        required_fields = ["top1", "topk", "meta"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Falta campo requerido: {field}")

        if "top1" in data:
            top1 = data["top1"]
            required_top1 = ["gloss", "confidence", "bucket", "is_other"]
            for field in required_top1:
                if field not in top1:
                    errors.append(f"top1 falta campo: {field}")

            # Validar tipos
            if "confidence" in top1 and not isinstance(top1["confidence"], (int, float)):
                errors.append("top1.confidence debe ser numérico")

            if "bucket" in top1 and top1["bucket"] not in ["HEAD", "MID", "OTHER"]:
                errors.append(f"top1.bucket inválido: {top1['bucket']}")

            if "is_other" in top1 and not isinstance(top1["is_other"], bool):
                errors.append("top1.is_other debe ser boolean")

        if "topk" in data:
            if not isinstance(data["topk"], list):
                errors.append("topk debe ser una lista")
            elif len(data["topk"]) == 0:
                errors.append("topk no puede estar vacío")
            else:
                # Validar primer elemento como ejemplo
                first_item = data["topk"][0]
                required_item = ["rank", "gloss", "confidence", "bucket"]
                for field in required_item:
                    if field not in first_item:
                        errors.append(f"topk[0] falta campo: {field}")

        if "meta" in data:
            meta = data["meta"]
            required_meta = ["model", "num_classes", "device"]
            for field in required_meta:
                if field not in meta:
                    errors.append(f"meta falta campo: {field}")

            if "num_classes" in meta and not isinstance(meta["num_classes"], int):
                errors.append("meta.num_classes debe ser integer")

        return errors

    def _validate_batch_schema(self, data: Dict[str, Any]) -> List[str]:
        """Valida esquema de respuesta de /infer/batch/evaluate"""
        errors = []

        required_fields = ["results", "errors", "sequence", "summary"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Falta campo requerido: {field}")

        if "results" in data and not isinstance(data["results"], list):
            errors.append("results debe ser una lista")

        if "sequence" in data:
            seq = data["sequence"]
            if "accepted" not in seq:
                errors.append("sequence falta campo 'accepted'")
            if "length" not in seq:
                errors.append("sequence falta campo 'length'")
            if "length" in seq and not isinstance(seq["length"], int):
                errors.append("sequence.length debe ser integer")

        if "summary" in data:
            summary = data["summary"]
            required_summary = ["total", "processed", "failed", "accepted", "rejected"]
            for field in required_summary:
                if field not in summary:
                    errors.append(f"summary falta campo: {field}")

        return errors

    def _validate_video_schema(self, data: Dict[str, Any]) -> List[str]:
        """Valida esquema de respuesta de /api/video/infer"""
        errors = []

        required_fields = ["results", "errors"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Falta campo requerido: {field}")

        if "results" in data:
            if not isinstance(data["results"], list):
                errors.append("results debe ser una lista")
            elif len(data["results"]) > 0:
                result = data["results"][0]
                required_result = ["video", "class_id", "class_name", "gloss", "score", "accepted", "reason"]
                for field in required_result:
                    if field not in result:
                        errors.append(f"results[0] falta campo: {field}")

                if "score" in result and not isinstance(result["score"], (int, float)):
                    errors.append("results[0].score debe ser numérico")

                if "accepted" in result and not isinstance(result["accepted"], bool):
                    errors.append("results[0].accepted debe ser boolean")

        return errors

    def _validate_health_schema(self, data: Dict[str, Any]) -> List[str]:
        """Valida esquema de respuesta de /health"""
        errors = []

        required_fields = ["status", "model_loaded"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Falta campo requerido: {field}")

        if "status" in data and data["status"] not in ["healthy", "unhealthy"]:
            errors.append(f"status inválido: {data['status']}")

        if "model_loaded" in data and not isinstance(data["model_loaded"], bool):
            errors.append("model_loaded debe ser boolean")

        if "model_loaded" in data and data["model_loaded"] and "num_classes" not in data:
            errors.append("num_classes requerido cuando model_loaded=true")

        return errors


def main():
    if len(sys.argv) != 3:
        print("Uso: python validate_responses.py <endpoint> <response_file.json>")
        print("\nEndpoints soportados:")
        print("  /infer")
        print("  /infer/batch/evaluate")
        print("  /api/video/infer")
        print("  /health")
        sys.exit(1)

    endpoint = sys.argv[1]
    response_file = Path(sys.argv[2])

    if not response_file.exists():
        print(f"Archivo no encontrado: {response_file}")
        sys.exit(1)

    try:
        with open(response_file, 'r', encoding='utf-8') as f:
            response_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error al parsear JSON: {e}")
        sys.exit(1)

    validator = ResponseValidator()
    errors = validator.validate_response(endpoint, response_data)

    if errors:
        print(f"❌ Validación FALLIDA para {endpoint}")
        print("Errores encontrados:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print(f"✅ Validación EXITOSA para {endpoint}")
        print("El esquema de respuesta cumple con las especificaciones.")


if __name__ == "__main__":
    main()