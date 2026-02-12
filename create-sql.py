
import sqlite3
import json
import os

# Nombre del archivo de base de datos
DB_NAME = "comsigns.db"

def create_database():
    # 1. Conectar (esto crea el archivo si no existe)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    print(f"🔧 Creando base de datos: {DB_NAME}...")

    # 2. Crear Tabla GLOSAS (Según tu documento de Sistemas de Info)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS glosas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gloss_id INTEGER UNIQUE NOT NULL,
        palabra TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 3. Crear Tabla LOGS (Para cumplir con el requisito de Auditoría/Mejora)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS inferencia_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input_type TEXT DEFAULT 'camera',
        prediccion TEXT,
        confianza REAL,
        latencia_ms INTEGER,
        fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 4. Poblar datos desde dict.json
    try:
        # Asegúrate de que dict.json esté en la misma carpeta
        with open("/Users/marloveper__/Documents/proyectos/COMSIGNS-BACKEND/comsigns-backend/models/dict.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            
        print(f"📂 Leyendo dict.json ({len(data)} entradas found)...")
        
        count = 0
        for gloss_id, values in data.items():
            # soportar dos formatos: {"gloss": "palabra", ...} o lista/tupla
            if isinstance(values, dict):
                palabra = values.get("gloss") or values.get("palabra") or ""
            elif isinstance(values, (list, tuple)) and len(values) > 0:
                palabra = values[0]
            else:
                palabra = str(values)

            # Normalizar gloss_id a entero cuando sea posible
            try:
                gid = int(gloss_id)
            except Exception:
                gid = gloss_id

            # Insertar ignorando duplicados (OR IGNORE)
            cursor.execute('INSERT OR IGNORE INTO glosas (gloss_id, palabra) VALUES (?, ?)', (gid, palabra))
            # Contar sólo inserciones reales
            if cursor.rowcount and cursor.rowcount > 0:
                count += 1
            
        conn.commit()
        print(f"✅ ¡Éxito! Se insertaron {count} glosas en la tabla 'glosas'.")
        
    except FileNotFoundError:
        print("⚠️  Error: No se encontró 'dict.json'. Asegúrate de ponerlo en la misma carpeta.")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_database()