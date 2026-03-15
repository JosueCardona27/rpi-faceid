"""
database.py
===========
Manejo de la base de datos SQLite para el sistema de acceso facial.

Cada persona almacena:
  - vector    : 413 floats (7 zonas × 59 bins LBP)
  - pesos     : 7 floats   (peso de cada zona al momento del registro)
  - dimensiones: 413

El reconocimiento usa distancia chi² ponderada por zonas,
lo que permite comparar caras parcialmente cubiertas.
"""

import sqlite3
import json
import numpy as np
from face_engine import distancia_ponderada, N_ZONAS, VECTOR_DIM

DB_PATH = "reconocimiento_facial.db"

# umbral de distancia chi² ponderada para dar acceso
# valores tipicos: 0.05-0.15 = muy parecido | >0.30 = diferente
UMBRAL = 0.12


def conectar():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ─── asegurar que la tabla tenga la columna pesos ─────────────────────────────
def _migrar_si_necesario():
    conn   = conectar()
    cols   = [r[1] for r in conn.execute(
        "PRAGMA table_info(vectores_faciales)").fetchall()]
    if "pesos" not in cols:
        conn.execute(
            "ALTER TABLE vectores_faciales ADD COLUMN pesos TEXT DEFAULT NULL")
        conn.commit()
        print("[DB] Columna 'pesos' agregada a vectores_faciales.")
    conn.close()

_migrar_si_necesario()


# ─── CRUD ─────────────────────────────────────────────────────────────────────

def registrar_persona(numero_cuenta: str, nombre_completo: str) -> int:
    try:
        conn   = conectar()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO personas (numero_cuenta, nombre_completo) VALUES (?, ?)",
            (numero_cuenta, nombre_completo)
        )
        conn.commit()
        pid = cursor.lastrowid
        print(f"[OK] Persona registrada ID={pid}: {nombre_completo}")
        return pid
    except sqlite3.IntegrityError:
        print(f"[AVISO] Cuenta '{numero_cuenta}' ya existe.")
        return -1
    finally:
        conn.close()


def guardar_vector_unico(persona_id: int,
                         vectores: list,
                         pesos_lista: list):
    """
    Recibe N vectores y N arrays de pesos, los promedia y guarda 1 solo registro.
    Los pesos promediados indican que zonas eran visibles durante el registro.
    """
    if not vectores:
        print("[ERROR] Lista de vectores vacia.")
        return

    v_prom = np.mean(vectores,     axis=0).astype(np.float32)
    p_prom = np.mean(pesos_lista,  axis=0).astype(np.float32)

    v_json = json.dumps(v_prom.tolist())
    p_json = json.dumps(p_prom.tolist())

    conn   = conectar()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO vectores_faciales
            (persona_id, vector, dimensiones, pesos)
        VALUES (?, ?, ?, ?)
    """, (persona_id, v_json, len(v_prom), p_json))
    conn.commit()
    conn.close()

    zonas_vis = int(np.sum(p_prom > 0.15))
    print(f"[OK] Vector ({len(v_prom)} dims, {zonas_vis}/7 zonas visibles) "
          f"guardado para ID {persona_id}.")


def cargar_todos_vectores() -> list:
    """
    Retorna lista de dicts con:
      persona_id, nombre, numero_cuenta, vector (np), pesos (np)
    """
    conn   = conectar()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.id, p.nombre_completo, p.numero_cuenta,
               vf.vector, vf.pesos
        FROM vectores_faciales vf
        JOIN personas p ON p.id = vf.persona_id
    """)
    rows = cursor.fetchall()
    conn.close()

    resultado = []
    for pid, nombre, cuenta, vjson, pjson in rows:
        v = np.array(json.loads(vjson), dtype=np.float32)
        p = (np.array(json.loads(pjson), dtype=np.float32)
             if pjson else np.ones(N_ZONAS, dtype=np.float32))
        resultado.append({
            "persona_id":    pid,
            "nombre":        nombre,
            "numero_cuenta": cuenta,
            "vector":        v,
            "pesos":         p,
        })
    return resultado


def reconocer_persona(vector_nuevo: np.ndarray,
                      pesos_nuevos: np.ndarray,
                      umbral: float = UMBRAL) -> dict | None:
    """
    Compara contra todos los registros usando distancia chi² ponderada por zonas.
    Siempre retorna el mejor candidato (con campo 'acceso': bool).
    Retorna None solo si la BD esta vacia.
    """
    personas = cargar_todos_vectores()
    if not personas:
        return None

    mejor     = None
    min_dist  = float("inf")
    min_zonas = 0

    for reg in personas:
        dist, n_zonas = distancia_ponderada(
            vector_nuevo, pesos_nuevos,
            reg["vector"],  reg["pesos"]
        )
        if dist < min_dist:
            min_dist  = dist
            min_zonas = n_zonas
            mejor     = reg.copy()
            mejor["distancia"]    = round(dist, 4)
            mejor["zonas_usadas"] = n_zonas

    if mejor is None:
        return None

    # convertir distancia a similitud %
    # dist=0 → 100%, dist=umbral → ~50%, dist>=0.5 → ~0%
    sim_raw = max(0.0, 1.0 - (mejor["distancia"] / 0.5))
    mejor["similitud_pct"] = round(sim_raw * 100, 1)
    mejor["acceso"]        = (mejor["distancia"] <= umbral and min_zonas >= 2)

    print(f"[INFO] Candidato: {mejor['nombre']} | "
          f"dist={mejor['distancia']:.4f} | "
          f"sim={mejor['similitud_pct']}% | "
          f"zonas={min_zonas} | "
          f"acceso={'SI' if mejor['acceso'] else 'NO'}")

    return mejor


def eliminar_persona(persona_id: int):
    conn = conectar()
    conn.execute("DELETE FROM personas WHERE id = ?", (persona_id,))
    conn.commit()
    conn.close()
    print(f"[DEL] Persona ID {persona_id} eliminada.")