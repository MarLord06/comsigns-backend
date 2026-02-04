# ComSigns Backend API

API de inferencia para lengua de señas LSP-AEC.

## 🚀 Deploy en Railway

### Requisitos previos
- Cuenta en [Railway](https://railway.app)
- Git instalado
- Repositorio conectado a Railway

### Variables de entorno (configurar en Railway Dashboard)

| Variable | Descripción | Default |
|----------|-------------|---------|
| `PORT` | Puerto del servidor (Railway lo asigna automáticamente) | 8000 |
| `COMSIGNS_DEVICE` | Dispositivo para inferencia | `cpu` |
| `LOG_LEVEL` | Nivel de logging | `INFO` |

### Pasos para deploy

1. **Conectar repositorio en Railway:**
   ```bash
   # Opción 1: Desde GitHub
   # Ve a Railway Dashboard → New Project → Deploy from GitHub repo
   
   # Opción 2: Usando Railway CLI
   npm i -g @railway/cli
   railway login
   railway init
   railway up
   ```

2. **Verificar el deploy:**
   - El build usa `nixpacks.toml` para instalar Python 3.11 + FFmpeg
   - El servidor inicia con `uvicorn` en el puerto asignado por Railway
   - Health check disponible en `/health`

## 📁 Estructura del proyecto

```
COMSIGNS-BACKEND/
├── railway.toml          # Config principal de Railway
├── nixpacks.toml         # Config de build (Python + FFmpeg)
├── runtime.txt           # Versión de Python
├── start.sh              # Script de inicio alternativo
├── .env.example          # Variables de entorno ejemplo
└── comsigns-backend/
    ├── requirements.txt  # Dependencias Python
    ├── backend/
    │   ├── api/          # FastAPI endpoints
    │   ├── services/     # Servicios de inferencia
    │   └── ...
    └── models/           # Modelos entrenados
```

## 🔧 Desarrollo local

```bash
# Clonar e instalar
cd comsigns-backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Ejecutar
uvicorn backend.api.app:app --reload --port 8000

# O usar el script
cd .. && ./start.sh
```

## 📡 Endpoints principales

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Info de la API |
| `/health` | GET | Health check |
| `/infer` | POST | Inferencia en archivo .pkl |
| `/api/video/infer` | POST | Inferencia en video |
| `/api/inference/batch` | POST | Inferencia batch con secuencia semántica |

## 🐛 Troubleshooting

### Build falla
- Verifica que `nixpacks.toml` esté en la raíz
- Revisa los logs de build en Railway Dashboard

### Health check falla
- El timeout está configurado a 300s para modelos grandes
- Verifica que el modelo `best.pt` esté incluido en el repo

### Error de módulos
- Asegúrate que `requirements.txt` esté en `comsigns-backend/`
- Verifica las rutas en `nixpacks.toml`
