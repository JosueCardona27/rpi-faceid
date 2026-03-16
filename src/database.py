"""
database.py
===========
Manejo de la base de datos SQLite para el sistema de acceso facial.

Cada persona almacena UN VECTOR POR ANGULO (paso de registro):
  - angulo    : str  — "frontal", "perfil_der", "perfil_izq", "abajo"
  - vector    : 413 floats (7 zonas x 59 bins LBP)
  - pesos     : 7 floats   (peso de cada zona)
  - n_muestras: cuantas muestras se promediaron para este angulo

El reconocimiento compara la cara actual contra TODOS los vectores
de TODAS las personas y retorna el mejor match por angulo.
Esto es mucho mas preciso que un unico vector promediado.
"""

import sqlite3
import json
import os
import numpy as np
from face_engine import distancia_ponderada, N_ZONAS, VECTOR_DIM

# ─── ruta de la base de datos ─────────────────────────────────────────────────
_DIR_ESTE_ARCHIVO = os.path.dirname(os.path.abspath(__file__))
_DIR_DB           = os.path.join(_DIR_ESTE_ARCHIVO, "..", "database")
os.makedirs(_DIR_DB, exist_ok=True)
DB_PATH = os.path.join(_DIR_DB, "reconocimiento_facial.db")

# umbral de distancia chi2 ponderada para dar acceso
UMBRAL = 0.12


def conectar():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _crear_tablas():
    """Crea todas las tablas necesarias si no existen."""
    conn = conectar()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS personas (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            numero_cuenta   TEXT    UNIQUE NOT NULL,
            nombre_completo TEXT    NOT NULL,
            fecha_registro  TEXT    DEFAULT (datetime('now'))
        );

        -- Tabla antigua: se mantiene para compatibilidad con registros previos
        CREATE TABLE IF NOT EXISTS vectores_faciales (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            persona_id  INTEGER UNIQUE NOT NULL,
            vector      TEXT    NOT NULL,
            dimensiones INTEGER NOT NULL,
            pesos       TEXT    DEFAULT NULL,
            FOREIGN KEY (persona_id) REFERENCES personas(id) ON DELETE CASCADE
        );

        -- Tabla nueva: un vector por angulo por persona
        CREATE TABLE IF NOT EXISTS vectores_por_angulo (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            persona_id  INTEGER NOT NULL,
            angulo      TEXT    NOT NULL,
            vector      TEXT    NOT NULL,
            pesos       TEXT    NOT NULL,
            n_muestras  INTEGER DEFAULT 1,
            UNIQUE(persona_id, angulo),
            FOREIGN KEY (persona_id) REFERENCES personas(id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    conn.close()

_crear_tablas()


# ─── migracion: asegurar columna pesos en tabla vieja ────────────────────────
def _migrar_si_necesario():
    conn = conectar()
    cols = [r[1] for r in conn.execute(
        "PRAGMA table_info(vectores_faciales)").fetchall()]
    if "pesos" not in cols:
        conn.execute(
            "ALTER TABLE vectores_faciales ADD COLUMN pesos TEXT DEFAULT NULL")
        conn.commit()
        print("[DB] Columna 'pesos' agregada a vectores_faciales.")
    conn.close()

_migrar_si_necesario()


# ─── CRUD personas ────────────────────────────────────────────────────────────

def registrar_persona(numero_cuenta: str, nombre_completo: str) -> int:
    try:
        conn   = conectar()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO personas (numero_cuenta, nombre_completo) VALUES (?, ?)",
            (numero_cuenta, nombre_completo))
        conn.commit()
        pid = cursor.lastrowid
        print(f"[OK] Persona registrada ID={pid}: {nombre_completo}")
        return pid
    except sqlite3.IntegrityError:
        print(f"[AVISO] Cuenta '{numero_cuenta}' ya existe.")
        return -1
    finally:
        conn.close()


# ─── NUEVO: guardar un vector por angulo ─────────────────────────────────────

def guardar_vectores_por_angulo(persona_id: int, vectores_por_paso: dict):
    """
    Guarda un vector promediado por cada angulo.

    vectores_por_paso formato:
    {
        "frontal":    {"vectores": [np.array, ...], "pesos": [np.array, ...]},
        "perfil_der": {"vectores": [...],           "pesos": [...]},
        "perfil_izq": {"vectores": [...],           "pesos": [...]},
        "abajo":      {"vectores": [...],           "pesos": [...]},
    }
    """
    conn   = conectar()
    cursor = conn.cursor()
    total  = 0

    for angulo, datos in vectores_por_paso.items():
        vecs  = datos.get("vectores", [])
        pesos = datos.get("pesos",    [])
        if not vecs:
            print(f"[WARN] Angulo '{angulo}' sin muestras, omitido.")
            continue

        v_prom = np.mean(vecs,  axis=0).astype(np.float32)
        p_prom = np.mean(pesos, axis=0).astype(np.float32)

        cursor.execute("""
            INSERT OR REPLACE INTO vectores_por_angulo
                (persona_id, angulo, vector, pesos, n_muestras)
            VALUES (?, ?, ?, ?, ?)
        """, (persona_id,
              angulo,
              json.dumps(v_prom.tolist()),
              json.dumps(p_prom.tolist()),
              len(vecs)))

        zonas_vis = int(np.sum(p_prom > 0.15))
        print(f"[OK] '{angulo}': {len(vecs)} muestras, "
              f"{zonas_vis}/7 zonas — ID {persona_id}")
        total += 1

    conn.commit()
    conn.close()
    print(f"[OK] {total} angulos guardados para persona ID {persona_id}.")


# ─── LEGACY: compatibilidad con codigo anterior ───────────────────────────────

def guardar_vector_unico(persona_id: int,
                          vectores: list,
                          pesos_lista: list):
    """Compatibilidad: promedia todo y guarda en tabla vieja."""
    if not vectores:
        print("[ERROR] Lista de vectores vacia.")
        return

    v_prom = np.mean(vectores,    axis=0).astype(np.float32)
    p_prom = np.mean(pesos_lista, axis=0).astype(np.float32)

    conn   = conectar()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO vectores_faciales
            (persona_id, vector, dimensiones, pesos)
        VALUES (?, ?, ?, ?)
    """, (persona_id,
          json.dumps(v_prom.tolist()),
          len(v_prom),
          json.dumps(p_prom.tolist())))
    conn.commit()
    conn.close()

    zonas_vis = int(np.sum(p_prom > 0.15))
    print(f"[OK] Vector legacy ({len(v_prom)} dims, "
          f"{zonas_vis}/7 zonas) guardado para ID {persona_id}.")


# ─── cargar todos los vectores (nuevo + legacy) ───────────────────────────────

def cargar_vectores_por_angulo() -> list:
    """
    Retorna todos los vectores registrados (tabla nueva + legacy).
    Formato de cada item:
      persona_id, nombre, numero_cuenta, angulo, vector, pesos, n_muestras
    """
    conn      = conectar()
    cursor    = conn.cursor()
    resultado = []

    # Registros nuevos
    cursor.execute("""
        SELECT p.id, p.nombre_completo, p.numero_cuenta,
               va.angulo, va.vector, va.pesos, va.n_muestras
        FROM vectores_por_angulo va
        JOIN personas p ON p.id = va.persona_id
    """)
    for pid, nombre, cuenta, angulo, vjson, pjson, n in cursor.fetchall():
        resultado.append({
            "persona_id":    pid,
            "nombre":        nombre,
            "numero_cuenta": cuenta,
            "angulo":        angulo,
            "vector":        np.array(json.loads(vjson), dtype=np.float32),
            "pesos":         np.array(json.loads(pjson), dtype=np.float32),
            "n_muestras":    n,
        })

    # Registros legacy — solo si esa persona NO tiene registros nuevos
    pids_nuevos = {r["persona_id"] for r in resultado}
    cursor.execute("""
        SELECT p.id, p.nombre_completo, p.numero_cuenta,
               vf.vector, vf.pesos
        FROM vectores_faciales vf
        JOIN personas p ON p.id = vf.persona_id
    """)
    for pid, nombre, cuenta, vjson, pjson in cursor.fetchall():
        if pid not in pids_nuevos:
            resultado.append({
                "persona_id":    pid,
                "nombre":        nombre,
                "numero_cuenta": cuenta,
                "angulo":        "legacy",
                "vector":        np.array(json.loads(vjson), dtype=np.float32),
                "pesos":         (np.array(json.loads(pjson), dtype=np.float32)
                                  if pjson else
                                  np.ones(N_ZONAS, dtype=np.float32)),
                "n_muestras":    1,
            })

    conn.close()
    return resultado


# ─── reconocimiento multi-angulo ─────────────────────────────────────────────

def reconocer_persona(vector_nuevo: np.ndarray,
                       pesos_nuevos: np.ndarray,
                       umbral: float = UMBRAL) -> dict | None:
    """
    Compara el vector nuevo contra TODOS los vectores por angulo
    de TODAS las personas. Para cada persona toma su mejor angulo
    y retorna la persona con menor distancia global.

    dist=0 → identico | dist>=0.5 → muy diferente
    """
    registros = cargar_vectores_por_angulo()
    if not registros:
        return None

    # Para cada persona guardar su mejor distancia entre todos sus angulos
    mejores = {}

    for reg in registros:
        pid        = reg["persona_id"]
        dist, nz   = distancia_ponderada(
            vector_nuevo, pesos_nuevos,
            reg["vector"], reg["pesos"])

        if pid not in mejores or dist < mejores[pid]["distancia"]:
            mejores[pid] = {
                "persona_id":    pid,
                "nombre":        reg["nombre"],
                "numero_cuenta": reg["numero_cuenta"],
                "angulo":        reg["angulo"],
                "distancia":     round(dist, 4),
                "zonas_usadas":  nz,
            }

    if not mejores:
        return None

    # La persona con menor distancia
    mejor = min(mejores.values(), key=lambda x: x["distancia"])

    sim_raw = max(0.0, 1.0 - (mejor["distancia"] / 0.5))
    mejor["similitud_pct"] = round(sim_raw * 100, 1)
    mejor["acceso"]        = (mejor["distancia"] <= umbral
                               and mejor["zonas_usadas"] >= 2)

    print(f"[INFO] {mejor['nombre']} | angulo={mejor['angulo']} | "
          f"dist={mejor['distancia']:.4f} | sim={mejor['similitud_pct']}% | "
          f"zonas={mejor['zonas_usadas']} | "
          f"acceso={'SI' if mejor['acceso'] else 'NO'}")

    return mejor


# ─── eliminar persona ─────────────────────────────────────────────────────────

def eliminar_persona(persona_id: int):
    conn = conectar()
    conn.execute("DELETE FROM personas WHERE id = ?", (persona_id,))
    conn.commit()
    conn.close()
    print(f"[DEL] Persona ID {persona_id} eliminada.")