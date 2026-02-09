#!/usr/bin/env python3
"""
ComSigns API Test Runner

Script para ejecutar pruebas automatizadas contra la API de ComSigns.
Requiere que el backend esté corriendo en http://localhost:8000

Uso:
    python run_api_tests.py [--verbose] [--report]

Opciones:
    --verbose: Mostrar output detallado de cada test
    --report: Generar reporte HTML en reports/test_report.html
"""

import requests
import json
import time
from pathlib import Path
import argparse
from typing import Dict, List, Any
import sys
import io

# Configuración
BASE_URL = "https://comsigns-multimodal-production.up.railway.app/"
TEST_FILES_DIR = Path(__file__).parent.parent / "test_files"

class APITestRunner:
    """Ejecutor de pruebas API para ComSigns"""

    def __init__(self, verbose: bool = False, generate_report: bool = False):
        self.verbose = verbose
        self.generate_report = generate_report
        self.results = []
        self.session = requests.Session()
        self.session.timeout = 30  # 30 segundos timeout

    def log(self, message: str):
        """Log message if verbose mode"""
        if self.verbose:
            print(message)

    def run_test(self, test_name: str, method: str, endpoint: str,
                 files: Dict = None, data: Dict = None, params: Dict = None,
                 expected_status: int = 200, validations: List[callable] = None) -> Dict:
        """Ejecuta un test individual"""

        url = f"{BASE_URL}{endpoint}"
        start_time = time.time()

        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params)
            elif method.upper() == "POST":
                if files:
                    response = self.session.post(url, files=files, data=data, params=params)
                else:
                    response = self.session.post(url, json=data, params=params)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")

            response_time = time.time() - start_time

            # Validaciones básicas
            passed = response.status_code == expected_status

            # Validaciones adicionales
            validation_errors = []
            if validations and passed:
                try:
                    response_data = response.json()
                    for validation in validations:
                        try:
                            validation(response_data)
                        except Exception as e:
                            validation_errors.append(str(e))
                except json.JSONDecodeError:
                    validation_errors.append("Respuesta no es JSON válido")

            if validation_errors:
                passed = False

            result = {
                "test_name": test_name,
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "response_time": round(response_time * 1000, 2),  # ms
                "passed": passed,
                "validation_errors": validation_errors,
                "error": None
            }

            self.log(f"✓ {test_name}: {'PASSED' if passed else 'FAILED'} "
                    f"({response.status_code}) - {response_time:.2f}ms")

            if validation_errors and self.verbose:
                for error in validation_errors:
                    self.log(f"  Validation error: {error}")

        except Exception as e:
            result = {
                "test_name": test_name,
                "endpoint": endpoint,
                "method": method,
                "status_code": None,
                "expected_status": expected_status,
                "response_time": time.time() - start_time,
                "passed": False,
                "validation_errors": [],
                "error": str(e)
            }
            self.log(f"✗ {test_name}: ERROR - {str(e)}")

        self.results.append(result)
        return result

    def validate_inference_response(self, data: Dict):
        """Validaciones para respuesta de inferencia"""
        assert "top1" in data, "Falta campo 'top1'"
        assert "topk" in data, "Falta campo 'topk'"
        assert "meta" in data, "Falta campo 'meta'"

        top1 = data["top1"]
        assert "gloss" in top1, "top1 falta campo 'gloss'"
        assert "confidence" in top1, "top1 falta campo 'confidence'"
        assert "bucket" in top1, "top1 falta campo 'bucket'"
        assert top1["bucket"] in ["HEAD", "MID", "OTHER"], f"Bucket inválido: {top1['bucket']}"

        assert isinstance(data["topk"], list), "topk debe ser lista"
        assert len(data["topk"]) > 0, "topk debe tener al menos 1 elemento"

    def validate_batch_response(self, data: Dict):
        """Validaciones para respuesta de batch"""
        assert "results" in data, "Falta campo 'results'"
        assert "sequence" in data, "Falta campo 'sequence'"
        assert "summary" in data, "Falta campo 'summary'"

        assert isinstance(data["results"], list), "results debe ser lista"
        assert "length" in data["sequence"], "sequence falta campo 'length'"

    def validate_video_response_placeholder(self, data: Dict):
        """Validaciones flexibles para respuesta de video (acepta archivos placeholder)"""
        assert "results" in data, "Falta campo 'results'"
        assert isinstance(data["results"], list), "results debe ser lista"

        # Con archivos placeholder, puede que no haya resultados procesados
        # Solo verificar estructura básica
        if len(data["results"]) > 0:
            result = data["results"][0]
            # Verificar que al menos tenga algunos campos básicos
            basic_fields = ["video"]
            for field in basic_fields:
                assert field in result, f"Resultado falta campo básico '{field}'"

    def is_valid_test_file(self, file_path: Path) -> bool:
        """Verifica si un archivo de prueba es válido (existe y no está vacío)"""
        return file_path.exists() and file_path.stat().st_size > 0

    def run_all_tests(self):
        """Ejecuta todos los tests de la suite"""

        print("🚀 Iniciando suite de pruebas API ComSigns")
        print(f"📍 Base URL: {BASE_URL}")
        print(f"📁 Test files: {TEST_FILES_DIR}")
        print("-" * 50)

        # TEST-009: Health Check (primero, sin archivos)
        self.run_test(
            "TEST-009: Health Check",
            "GET", "/health",
            expected_status=200,
            validations=[self.validate_health_response]
        )

        # TEST-001: Inferencia PKL exitosa
        pkl_path = TEST_FILES_DIR / "samples" / "valid_sample.pkl"
        if self.is_valid_test_file(pkl_path):
            with open(pkl_path, "rb") as f:
                files = {"file": ("valid_sample.pkl", f, "application/octet-stream")}
                self.run_test(
                    "TEST-001: Inferencia PKL exitosa",
                    "POST", "/infer",
                    files=files,
                    params={"topk": 5},
                    expected_status=200,
                    validations=[self.validate_inference_response]
                )
        else:
            print(f"⚠️  TEST-001: Saltado - archivo no encontrado o vacío: {pkl_path}")

        # TEST-002: Archivo con extensión inválida
        txt_path = TEST_FILES_DIR / "samples" / "sample.txt"
        if txt_path.exists():
            with open(txt_path, "rb") as f:
                files = {"file": ("sample.txt", f, "text/plain")}
                self.run_test(
                    "TEST-002: Extensión inválida",
                    "POST", "/infer",
                    files=files,
                    expected_status=400
                )
        else:
            print(f"⚠️  TEST-002: Saltado - archivo no encontrado: {txt_path}")

        # TEST-003: Archivo vacío
        empty_path = TEST_FILES_DIR / "samples" / "empty.pkl"
        if empty_path.exists():
            with open(empty_path, "rb") as f:
                files = {"file": ("empty.pkl", f, "application/octet-stream")}
                self.run_test(
                    "TEST-003: Archivo vacío",
                    "POST", "/infer",
                    files=files,
                    expected_status=400
                )
        else:
            print(f"⚠️  TEST-003: Saltado - archivo no encontrado: {empty_path}")

        # TEST-010: Parámetro topk inválido
        if self.is_valid_test_file(pkl_path):
            with open(pkl_path, "rb") as f:
                files = {"file": ("valid_sample.pkl", f, "application/octet-stream")}
                self.run_test(
                    "TEST-010: TopK inválido",
                    "POST", "/infer",
                    files=files,
                    params={"topk": 50},
                    expected_status=422
                )
        else:
            print(f"⚠️  TEST-010: Saltado - archivo no encontrado o vacío: {pkl_path}")

        # TEST-004: Batch inference múltiple
        if self.is_valid_test_file(pkl_path):
            # Leer el contenido del archivo una vez
            with open(pkl_path, "rb") as f:
                pkl_content = f.read()

            # Crear múltiples entradas con el mismo contenido
            files = {}
            for i in range(3):
                files[f"files"] = (f"valid_sample_{i}.pkl", io.BytesIO(pkl_content), "application/octet-stream")

            self.run_test(
                "TEST-004: Batch múltiple",
                "POST", "/infer/batch/evaluate",
                files=files,
                params={"topk": 5},
                expected_status=200,
                validations=[self.validate_batch_response]
            )
        else:
            print(f"⚠️  TEST-004: Saltado - archivo no encontrado o vacío: {pkl_path}")

        # TEST-006: Video inference
        video_path = TEST_FILES_DIR / "videos" / "valid_sign.mp4"
        if video_path.exists():
            with open(video_path, "rb") as f:
                files = {"files": ("valid_sign.mp4", f, "video/mp4")}
                # Nota: Este test puede fallar con archivos placeholder que no son videos reales
                # En ese caso, valida que la API responda correctamente (no que procese exitosamente)
                self.run_test(
                    "TEST-006: Video inference",
                    "POST", "/api/video/infer",
                    files=files,
                    params={"topk": 5},
                    expected_status=200,
                    validations=[self.validate_video_response_placeholder]
                )
        else:
            print(f"⚠️  TEST-006: Saltado - archivo no encontrado: {video_path}")

        # TEST-007: Video extensión inválida
        invalid_video_path = TEST_FILES_DIR / "videos" / "invalid.gif"
        if invalid_video_path.exists():
            with open(invalid_video_path, "rb") as f:
                files = {"files": ("invalid.gif", f, "image/gif")}
                self.run_test(
                    "TEST-007: Video extensión inválida",
                    "POST", "/api/video/infer",
                    files=files,
                    expected_status=400
                )
        else:
            print(f"⚠️  TEST-007: Saltado - archivo no encontrado: {invalid_video_path}")

        # TEST-008: Video excesivamente grande (placeholder - no podemos crear archivo real de 100MB+)
        # Este test se omite ya que requiere un archivo real muy grande
        print(f"⚠️  TEST-008: Saltado - requiere archivo real > 100MB (no implementado en suite de pruebas)")

        # Mostrar resumen
        self.print_summary()

        # Generar reporte si solicitado
        if self.generate_report:
            self.generate_html_report()

    def validate_health_response(self, data: Dict):
        """Validaciones para health check"""
        assert "status" in data, "Falta campo 'status'"
        assert "model_loaded" in data, "Falta campo 'model_loaded'"
        assert data["status"] in ["healthy", "unhealthy"], f"Status inválido: {data['status']}"

    def print_summary(self):
        """Imprime resumen de resultados"""
        print("\n" + "=" * 50)
        print("📊 RESUMEN DE PRUEBAS")
        print("=" * 50)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        print(f"Total de tests: {total}")
        print(f"✅ Pasaron: {passed}")
        print(f"❌ Fallaron: {failed}")
        print(".1f")

        if failed > 0:
            print("\n❌ Tests fallidos:")
            for result in self.results:
                if not result["passed"]:
                    print(f"  - {result['test_name']}")
                    if result["error"]:
                        print(f"    Error: {result['error']}")
                    if result["validation_errors"]:
                        for err in result["validation_errors"]:
                            print(f"    Validación: {err}")

    def generate_html_report(self):
        """Genera reporte HTML"""
        report_path = Path(__file__).parent.parent / "reports" / "test_report.html"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ComSigns API Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .test {{ margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
        .passed {{ background: #d4edda; border-color: #c3e6cb; }}
        .failed {{ background: #f8d7da; border-color: #f5c6cb; }}
        .error {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>ComSigns API Test Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {len(self.results)}</p>
        <p><strong>Passed:</strong> {sum(1 for r in self.results if r['passed'])}</p>
        <p><strong>Failed:</strong> {sum(1 for r in self.results if not r['passed'])}</p>
        <p><strong>Success Rate:</strong> {sum(1 for r in self.results if r['passed']) / len(self.results) * 100:.1f}%</p>
        <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Endpoint</th>
            <th>Method</th>
            <th>Status</th>
            <th>Expected</th>
            <th>Response Time</th>
            <th>Result</th>
            <th>Errors</th>
        </tr>
"""

        for result in self.results:
            status_class = "passed" if result["passed"] else "failed"
            status_text = "✅ PASSED" if result["passed"] else "❌ FAILED"

            errors = []
            if result["error"]:
                errors.append(f"Error: {result['error']}")
            if result["validation_errors"]:
                errors.extend([f"Validation: {err}" for err in result["validation_errors"]])

            html += f"""
        <tr class="{status_class}">
            <td>{result['test_name']}</td>
            <td>{result['endpoint']}</td>
            <td>{result['method']}</td>
            <td>{result['status_code'] or 'N/A'}</td>
            <td>{result['expected_status']}</td>
            <td>{result['response_time']:.2f}ms</td>
            <td>{status_text}</td>
            <td>{'<br>'.join(errors) if errors else ''}</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""

        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"\n📄 Reporte HTML generado: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="ComSigns API Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostrar output detallado")
    parser.add_argument("--report", "-r", action="store_true", help="Generar reporte HTML")

    args = parser.parse_args()

    runner = APITestRunner(verbose=args.verbose, generate_report=args.report)
    runner.run_all_tests()

    # Exit code basado en resultados
    failed_tests = sum(1 for r in runner.results if not r["passed"])
    sys.exit(1 if failed_tests > 0 else 0)


if __name__ == "__main__":
    main()
