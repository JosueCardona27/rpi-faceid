"""
database.py
===========
Base de datos SQLite para el sistema de acceso facial.

CAMBIOS:
  - numero_cuenta ya NO es obligatorio (puede ser NULL)
  - UMBRAL: 2.2  (LBP necesita tolerancia)
  - MAX_DIST: 10.0
"""

import sqlite3
import json
import os
import numpy as np
from face_engine import distancia_chi2, VECTOR_DIM

_DIR_ESTE_ARCHIVO = os.path.dirname(os.path.abspath(__file__))
_DIR_DB           = os.path.join(_DIR_ESTE_ARCHIVO, "..", "database")
os.makedirs(_DIR_DB, exist_ok=True)
DB_PATH = os.path.join(_DIR_DB, "reconocimiento_facial.db")

UMBRAL        = 1.0   # Acceso permitido si distancia <= 1.0
UMBRAL_RECHAZO = 2.0   # Si distancia > 2.0 → persona desconocida (retorna None)
MAX_DIST      = 10.0

ROLES_VALIDOS = ("admin", "maestro", "estudiante")


def conectar():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _crear_tablas():
    conn = conectar()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            id_usuario       TEXT    NOT NULL UNIQUE,
            nombre           TEXT    NOT NULL,
            apellido_paterno TEXT    NOT NULL,
            apellido_materno TEXT    NOT NULL DEFAULT '.',
            numero_cuenta    TEXT,
            rol              TEXT    NOT NULL
                             CHECK(rol IN ('admin','maestro','estudiante')),
            fecha_registro   TEXT    DEFAULT (datetime('now','localtime')),
            registrado_por   INTEGER
        );

        CREATE TABLE IF NOT EXISTS vectores_faciales (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id    INTEGER NOT NULL UNIQUE,
            vector        TEXT    NOT NULL,
            dimensiones   INTEGER DEFAULT 512,
            fecha_captura TEXT    DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS vectores_por_angulo (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id  INTEGER NOT NULL,
            angulo      TEXT    NOT NULL,
            vector      TEXT    NOT NULL,
            n_muestras  INTEGER DEFAULT 1,
            UNIQUE(usuario_id, angulo),
            FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usuarios_rol ON usuarios(rol)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_vf_usuario ON vectores_faciales(usuario_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vpa_usuario ON vectores_por_angulo(usuario_id)")
    conn.commit()
    conn.close()

_crear_tablas()


# =============================================================================
#  CRUD USUARIOS
# =============================================================================

def registrar_usuario(
        nombre:           str,
        apellido_paterno: str,
        apellido_materno: str = "",
        rol:              str = "estudiante",
        numero_cuenta:    str = None,
        telefono:         int = None,
        registrado_por:   int = None,
) -> int:
    if rol not in ROLES_VALIDOS:
        print(f"[ERROR] Rol '{rol}' no valido.")
        return -1

    # Generar id_usuario unico sin depender de numero_cuenta
    import time as _time
    sufijo     = numero_cuenta if numero_cuenta else str(int(_time.time() * 1000))[-8:]
    id_usuario = f"{rol[:3].upper()}-{sufijo}"
    ap_mat     = apellido_materno.strip() if apellido_materno.strip() else "."

    try:
        conn   = conectar()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO usuarios
                (id_usuario, nombre, apellido_paterno, apellido_materno,
                 numero_cuenta, rol, registrado_por)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (id_usuario, nombre, apellido_paterno, ap_mat,
              numero_cuenta, rol, registrado_por))
        conn.commit()
        uid = cursor.lastrowid
        print(f"[DB] Usuario registrado ID={uid}: {nombre} {apellido_paterno} ({rol})")
        return uid
    except sqlite3.IntegrityError as e:
        print(f"[DB] Error de integridad: {e}")
        return -1
    finally:
        conn.close()


def registrar_persona(nombre_completo: str, numero_cuenta: str = None) -> int:
    partes = nombre_completo.strip().split()
    if len(partes) >= 3:
        nombre = " ".join(partes[:-2]); ap_pat = partes[-2]; ap_mat = partes[-1]
    elif len(partes) == 2:
        nombre = partes[0]; ap_pat = partes[1]; ap_mat = ""
    else:
        nombre = nombre_completo.strip(); ap_pat = "."; ap_mat = ""
    return registrar_usuario(nombre, ap_pat, ap_mat, numero_cuenta=numero_cuenta)


# =============================================================================
#  GUARDAR VECTORES POR ANGULO
# =============================================================================

def guardar_vectores_por_angulo(usuario_id: int, vectores_por_paso: dict):
    conn   = conectar()
    cursor = conn.cursor()
    total  = 0

    for angulo, datos in vectores_por_paso.items():
        vecs = datos.get("vectores", [])
        if not vecs:
            print(f"[DB] Angulo '{angulo}' sin muestras.")
            continue

        v_prom = np.mean(vecs, axis=0).astype(np.float32)
        if len(v_prom) != VECTOR_DIM:
            print(f"[DB] Angulo '{angulo}': {len(v_prom)} dims != {VECTOR_DIM}. Omitido.")
            continue

        cursor.execute("""
            INSERT OR REPLACE INTO vectores_por_angulo
                (usuario_id, angulo, vector, n_muestras)
            VALUES (?, ?, ?, ?)
        """, (usuario_id, angulo, json.dumps(v_prom.tolist()), len(vecs)))
        print(f"[DB] ID={usuario_id} angulo='{angulo}' muestras={len(vecs)}")
        total += 1

    conn.commit()
    conn.close()
    print(f"[DB] {total} angulos guardados para usuario ID={usuario_id}.")


def guardar_vector_unico(usuario_id: int, vectores: list, pesos_lista=None):
    if not vectores:
        return
    v_prom = np.mean(vectores, axis=0).astype(np.float32)
    conn   = conectar()
    conn.execute("""
        INSERT OR REPLACE INTO vectores_faciales (usuario_id, vector, dimensiones)
        VALUES (?, ?, ?)
    """, (usuario_id, json.dumps(v_prom.tolist()), len(v_prom)))
    conn.commit()
    conn.close()


# =============================================================================
#  CARGAR VECTORES
# =============================================================================

def cargar_vectores_por_angulo() -> list:
    conn      = conectar()
    cursor    = conn.cursor()
    resultado = []

    cursor.execute("""
        SELECT u.id,
               u.nombre || ' ' || u.apellido_paterno || ' ' || u.apellido_materno,
               u.numero_cuenta, u.rol,
               va.angulo, va.vector, va.n_muestras
        FROM vectores_por_angulo va
        JOIN usuarios u ON u.id = va.usuario_id
        ORDER BY u.id, va.angulo
    """)
    for uid, nombre, cuenta, rol, angulo, vjson, n in cursor.fetchall():
        v = np.array(json.loads(vjson), dtype=np.float32)
        if len(v) == VECTOR_DIM:
            resultado.append({
                "usuario_id":    uid,
                "nombre":        nombre.strip(),
                "numero_cuenta": cuenta or "",
                "rol":           rol,
                "angulo":        angulo,
                "vector":        v,
                "n_muestras":    n,
            })

    ids_nuevos = {r["usuario_id"] for r in resultado}

    cursor.execute("""
        SELECT u.id,
               u.nombre || ' ' || u.apellido_paterno || ' ' || u.apellido_materno,
               u.numero_cuenta, u.rol, vf.vector, vf.dimensiones
        FROM vectores_faciales vf
        JOIN usuarios u ON u.id = vf.usuario_id
    """)
    for uid, nombre, cuenta, rol, vjson, dims in cursor.fetchall():
        if uid in ids_nuevos or dims != VECTOR_DIM:
            continue
        v = np.array(json.loads(vjson), dtype=np.float32)
        resultado.append({
            "usuario_id":    uid,
            "nombre":        nombre.strip(),
            "numero_cuenta": cuenta or "",
            "rol":           rol,
            "angulo":        "legacy",
            "vector":        v,
            "n_muestras":    1,
        })

    conn.close()
    return resultado


# =============================================================================
#  RECONOCIMIENTO
# =============================================================================

def reconocer_persona(vector_nuevo: np.ndarray,
                       umbral: float = UMBRAL) -> dict | None:
    """
    Compara vector_nuevo contra todos los registrados.

    Retorna None en dos casos:
      1. No hay registros en la base de datos.
      2. La distancia minima supera UMBRAL_RECHAZO — significa que la
         persona escaneada no esta en la base de datos (desconocida).
         Esto evita que siempre se devuelva "el mas cercano" aunque sea
         una persona completamente diferente.
    """
    registros = cargar_vectores_por_angulo()
    if not registros:
        return None

    mejores: dict[int, dict] = {}
    for reg in registros:
        uid  = reg["usuario_id"]
        dist = distancia_chi2(vector_nuevo, reg["vector"])
        if uid not in mejores or dist < mejores[uid]["distancia"]:
            mejores[uid] = {
                "usuario_id":    uid,
                "nombre":        reg["nombre"],
                "numero_cuenta": reg["numero_cuenta"],
                "rol":           reg["rol"],
                "angulo":        reg["angulo"],
                "distancia":     round(dist, 4),
            }

    if not mejores:
        return None

    mejor = min(mejores.values(), key=lambda x: x["distancia"])

    # Si la mejor distancia supera UMBRAL_RECHAZO, la persona no esta
    # registrada — retornar None en lugar de dar un falso positivo.
    if mejor["distancia"] > UMBRAL_RECHAZO:
        print(f"[RECONO] DESCONOCIDO | mejor candidato: {mejor['nombre']} | "
              f"dist={mejor['distancia']:.4f} > rechazo={UMBRAL_RECHAZO}")
        return None

    sim_raw     = max(0.0, 1.0 - (mejor["distancia"] / MAX_DIST))
    mejor["similitud_pct"] = round(sim_raw * 100, 1)
    mejor["acceso"]        = mejor["distancia"] <= umbral

    print(f"[RECONO] {mejor['nombre']} | angulo={mejor['angulo']} | "
          f"dist={mejor['distancia']:.4f} | sim={mejor['similitud_pct']}% | "
          f"acceso={'SI' if mejor['acceso'] else 'NO'}")
    return mejor


# =============================================================================
#  ELIMINAR / CONSULTAS
# =============================================================================

def eliminar_persona(usuario_id: int):
    conn = conectar()
    conn.execute("DELETE FROM usuarios WHERE id = ?", (usuario_id,))
    conn.commit()
    conn.close()
    print(f"[DB] Usuario ID={usuario_id} eliminado.")

eliminar_usuario = eliminar_persona


def listar_usuarios(rol: str = None) -> list:
    conn   = conectar()
    cursor = conn.cursor()
    if rol:
        cursor.execute(
            "SELECT id, id_usuario, nombre, apellido_paterno, apellido_materno, "
            "numero_cuenta, rol FROM usuarios WHERE rol=? ORDER BY apellido_paterno",
            (rol,))
    else:
        cursor.execute(
            "SELECT id, id_usuario, nombre, apellido_paterno, apellido_materno, "
            "numero_cuenta, rol FROM usuarios ORDER BY apellido_paterno")
    rows = cursor.fetchall()
    conn.close()
    return rows


def tiene_vector(usuario_id: int) -> bool:
    conn = conectar()
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM vectores_por_angulo WHERE usuario_id=?", (usuario_id,))
    n1 = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM vectores_faciales WHERE usuario_id=?", (usuario_id,))
    n2 = c.fetchone()[0]
    conn.close()
    return (n1 + n2) > 0


def angulos_registrados(usuario_id: int) -> list:
    conn = conectar()
    c    = conn.cursor()
    c.execute("SELECT angulo, n_muestras FROM vectores_por_angulo "
              "WHERE usuario_id=? ORDER BY angulo", (usuario_id,))
    rows = c.fetchall()
    conn.close()
    return rows