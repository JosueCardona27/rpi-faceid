"""
database.py
===========
Base de datos SQLite para el sistema de reconocimiento facial.

Cambios v4:
  - contrasena obligatoria para admin y maestro (hash SHA-256)
  - correo obligatorio para admin y maestro
  - estudiantes NO tienen correo ni contrasena
  - grado y grupo SOLO para estudiantes

Umbrales calibrados:
  UMBRAL  = 0.55   acceso permitido
  RECHAZO = 0.58   desconocido
"""

import sqlite3
import hashlib
import json
import os
import sys
import re
import numpy as np
from face_engine import distancia_chi2, VECTOR_DIM

_DIR_ESTE_ARCHIVO = os.path.dirname(os.path.abspath(__file__))
_DIR_DB           = os.path.join(_DIR_ESTE_ARCHIVO, "..", "database")
DB_PATH           = os.path.join(_DIR_DB, "reconocimiento_facial.db")

# =============================================================================
#  CONSTANTES
# =============================================================================

UMBRAL           = 0.55
RECHAZO          = 0.58
GAP_MIN          = 0.05
MAX_DIST         = 1.0
UMBRAL_DUPLICADO = 0.20

ANGULOS_VALIDOS = ("frontal", "perfil_der", "perfil_izq")
ROLES_VALIDOS   = ("admin", "maestro", "estudiante")

PERMISOS_REGISTRO = {
    "admin":      ("admin", "maestro", "estudiante"),
    "maestro":    ("estudiante",),
    "estudiante": (),
}

GRADO_MIN = 1
GRADO_MAX = 20


# =============================================================================
#  VERIFICACION DE BASE DE DATOS AL INICIO
# =============================================================================

def _verificar_bd():
    if not os.path.exists(DB_PATH):
        print()
        print("=" * 60)
        print("  [AVISO] Base de datos no encontrada.")
        print(f"  Ruta esperada: {DB_PATH}")
        print("  Las funciones de BD estarán deshabilitadas.")
        print("=" * 60)
        return   # <-- antes era sys.exit(1)

    tablas_requeridas = {
        "usuarios", "estudiantes_detalle",
        "vectores_por_angulo", "registro_acceso", "auditoria_cambios",
    }
    try:
        conn   = sqlite3.connect(DB_PATH)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")
        tablas = {row[0] for row in cursor.fetchall()}
        conn.close()
    except sqlite3.Error as e:
        print(f"\n[AVISO] No se pudo abrir la BD: {e}\n")
        return   # <-- antes era sys.exit(1)

    faltantes = tablas_requeridas - tablas
    if faltantes:
        print(f"\n[AVISO] BD incompleta. Tablas faltantes: "
              f"{', '.join(sorted(faltantes))}\n")
        return   # <-- antes era sys.exit(1)

    print(f"[DB] BD verificada: {DB_PATH}")

_verificar_bd()


# =============================================================================
#  CONEXION
# =============================================================================

def conectar() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode  = WAL")
    conn.execute("PRAGMA synchronous   = NORMAL")
    return conn


def _tablas_existentes(conn) -> set:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cur.fetchall()}


# =============================================================================
#  HASH DE CONTRASENA
# =============================================================================

def hash_contrasena(contrasena: str) -> str:
    """Devuelve el hash SHA-256 de la contrasena en hexadecimal."""
    return hashlib.sha256(contrasena.encode("utf-8")).hexdigest()


# =============================================================================
#  VALIDACIONES
# =============================================================================

def validar_numero_cuenta(numero_cuenta: str) -> tuple[bool, str]:
    if not numero_cuenta:
        return False, "El numero de cuenta es obligatorio."
    if not re.fullmatch(r"\d{8}", numero_cuenta.strip()):
        return False, "El numero de cuenta debe tener exactamente 8 digitos."
    return True, ""


def validar_correo(correo: str) -> tuple[bool, str]:
    if not correo or not correo.strip():
        return False, "El correo es obligatorio para admin y maestro."
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", correo.strip()):
        return False, "El correo no tiene un formato valido."
    return True, ""


def validar_contrasena(contrasena: str) -> tuple[bool, str]:
    if not contrasena or len(contrasena.strip()) < 6:
        return False, "La contrasena debe tener al menos 6 caracteres."
    return True, ""


def validar_grado(grado) -> tuple[bool, str]:
    try:
        g = int(grado)
        if not (GRADO_MIN <= g <= GRADO_MAX):
            return False, f"El grado debe estar entre {GRADO_MIN} y {GRADO_MAX}."
        return True, ""
    except (ValueError, TypeError):
        return False, "El grado debe ser un numero entero."


def validar_grupo(grupo: str) -> tuple[bool, str]:
    if not grupo:
        return False, "El grupo es obligatorio para estudiantes."
    if not re.fullmatch(r"[A-Za-z]", grupo.strip()):
        return False, "El grupo debe ser una sola letra (A-Z)."
    return True, ""


# =============================================================================
#  PERMISOS
# =============================================================================

def puede_registrar(rol_registrador: str, rol_nuevo: str) -> bool:
    return rol_nuevo in PERMISOS_REGISTRO.get(rol_registrador, ())


# =============================================================================
#  CRUD USUARIOS
# =============================================================================

def registrar_usuario(
        nombre:           str,
        apellido_paterno: str,
        apellido_materno: str = "",
        rol:              str = "estudiante",
        numero_cuenta:    str = None,
        correo:           str = None,
        contrasena:       str = None,   # texto plano — se hashea aqui
        grado:            int = None,
        grupo:            str = None,
        registrado_por:   int = None,
        rol_registrador:  str = None,
        **kwargs,
) -> int:
    """
    Registra un nuevo usuario.

    Admin/Maestro: requieren correo y contrasena.
    Estudiante:    requieren grado y grupo. NO pueden tener correo/contrasena.

    La contrasena se recibe en texto plano y se almacena como SHA-256.
    """
    if rol not in ROLES_VALIDOS:
        print(f"[ERROR] Rol '{rol}' no valido.")
        return -1

    if rol_registrador is not None:
        if not puede_registrar(rol_registrador, rol):
            print(f"[ERROR] '{rol_registrador}' no puede registrar '{rol}'.")
            return -1

    ok, msg = validar_numero_cuenta(numero_cuenta)
    if not ok:
        print(f"[ERROR] {msg}"); return -1

    if rol in ("admin", "maestro"):
        ok, msg = validar_correo(correo)
        if not ok:
            print(f"[ERROR] {msg}"); return -1
        ok, msg = validar_contrasena(contrasena)
        if not ok:
            print(f"[ERROR] {msg}"); return -1

    if rol == "estudiante":
        ok, msg = validar_grado(grado)
        if not ok:
            print(f"[ERROR] {msg}"); return -1
        ok, msg = validar_grupo(grupo)
        if not ok:
            print(f"[ERROR] {msg}"); return -1

    ap_mat      = apellido_materno.strip() if apellido_materno.strip() else "."
    hash_pwd    = hash_contrasena(contrasena) if contrasena else None
    correo_limpio = correo.strip().lower() if correo else None

    try:
        conn   = conectar()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO usuarios
                (nombre, apellido_paterno, apellido_materno,
                 numero_cuenta, correo, contrasena, rol, registrado_por)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (nombre, apellido_paterno, ap_mat,
              numero_cuenta.strip(), correo_limpio,
              hash_pwd, rol, registrado_por))

        uid = cursor.lastrowid

        if rol == "estudiante":
            cursor.execute("""
                INSERT INTO estudiantes_detalle (usuario_id, grado, grupo)
                VALUES (?, ?, ?)
            """, (uid, int(grado), grupo.strip().upper()))

        conn.commit()
        print(f"[DB] Usuario registrado ID={uid}: "
              f"{nombre} {apellido_paterno} ({rol})")
        return uid

    except sqlite3.IntegrityError as e:
        print(f"[DB] Error de integridad: {e}")
        return -1
    finally:
        conn.close()


def eliminar_usuario(usuario_id: int) -> bool:
    try:
        conn = conectar()
        conn.execute("DELETE FROM usuarios WHERE id = ?", (usuario_id,))
        conn.commit()
        conn.close()
        print(f"[DB] Usuario ID={usuario_id} eliminado.")
        return True
    except sqlite3.Error as e:
        print(f"[DB] Error al eliminar: {e}")
        return False

eliminar_persona = eliminar_usuario


def listar_usuarios(rol: str = None) -> list:
    conn   = conectar()
    cursor = conn.cursor()
    if rol:
        cursor.execute("""
            SELECT id, nombre, apellido_paterno, apellido_materno,
                   numero_cuenta, correo, rol, grado, grupo, angulos_registrados
            FROM vista_usuarios WHERE rol = ?
            ORDER BY apellido_paterno, nombre
        """, (rol,))
    else:
        cursor.execute("""
            SELECT id, nombre, apellido_paterno, apellido_materno,
                   numero_cuenta, correo, rol, grado, grupo, angulos_registrados
            FROM vista_usuarios
            ORDER BY apellido_paterno, nombre
        """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def obtener_usuario(usuario_id: int) -> dict | None:
    conn   = conectar()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, nombre, apellido_paterno, apellido_materno,
               numero_cuenta, correo, rol, grado, grupo,
               registrado_por, fecha_registro
        FROM vista_usuarios WHERE id = ?
    """, (usuario_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    keys = ("id","nombre","apellido_paterno","apellido_materno",
            "numero_cuenta","correo","rol","grado","grupo",
            "registrado_por","fecha_registro")
    return dict(zip(keys, row))


def registrar_persona(nombre_completo: str,
                       numero_cuenta: str = None) -> int:
    partes = nombre_completo.strip().split()
    if len(partes) >= 3:
        nombre = " ".join(partes[:-2]); ap_pat = partes[-2]; ap_mat = partes[-1]
    elif len(partes) == 2:
        nombre = partes[0]; ap_pat = partes[1]; ap_mat = ""
    else:
        nombre = nombre_completo.strip(); ap_pat = "."; ap_mat = ""
    return registrar_usuario(nombre, ap_pat, ap_mat,
                             numero_cuenta=numero_cuenta)


# =============================================================================
#  GUARDAR / CARGAR VECTORES
# =============================================================================

def guardar_vectores_por_angulo(usuario_id: int,
                                 vectores_por_paso: dict) -> int:
    conn   = conectar()
    cursor = conn.cursor()
    total  = 0
    for angulo, datos in vectores_por_paso.items():
        if angulo not in ANGULOS_VALIDOS:
            continue
        vecs = datos.get("vectores", [])
        if not vecs:
            continue
        v_prom = np.mean(vecs, axis=0).astype(np.float32)
        if len(v_prom) != VECTOR_DIM:
            continue
        cursor.execute("""
            INSERT OR REPLACE INTO vectores_por_angulo
                (usuario_id, angulo, vector, dimensiones, n_muestras)
            VALUES (?, ?, ?, ?, ?)
        """, (usuario_id, angulo,
              json.dumps(v_prom.tolist()), VECTOR_DIM, len(vecs)))
        print(f"[DB] ID={usuario_id} angulo='{angulo}' muestras={len(vecs)}")
        total += 1
    conn.commit()
    conn.close()
    print(f"[DB] {total} angulos guardados para ID={usuario_id}.")
    return total


def guardar_vector_unico(usuario_id: int, vectores: list,
                          pesos_lista=None):
    if not vectores:
        return
    v_prom = np.mean(vectores, axis=0).astype(np.float32)
    guardar_vectores_por_angulo(
        usuario_id, {"frontal": {"vectores": [v_prom]}})


def cargar_vectores_por_angulo(excluir_id: int = None) -> list:
    conn      = conectar()
    cursor    = conn.cursor()
    resultado = []
    cursor.execute("""
        SELECT u.id,
               u.nombre||' '||u.apellido_paterno||' '||u.apellido_materno,
               u.numero_cuenta, u.rol,
               va.angulo, va.vector, va.n_muestras, va.dimensiones
        FROM vectores_por_angulo va
        JOIN usuarios u ON u.id = va.usuario_id
        ORDER BY u.id, va.angulo
    """)
    for uid, nombre, cuenta, rol, angulo, vjson, n, dims in cursor.fetchall():
        if excluir_id is not None and uid == excluir_id:
            continue
        if dims != VECTOR_DIM:
            continue
        v = np.array(json.loads(vjson), dtype=np.float32)
        resultado.append({
            "usuario_id":    uid,
            "nombre":        nombre.strip(),
            "numero_cuenta": cuenta or "",
            "rol":           rol,
            "angulo":        angulo,
            "vector":        v,
            "n_muestras":    n,
        })
    conn.close()
    return resultado


# =============================================================================
#  RECONOCIMIENTO
# =============================================================================

def reconocer_persona(
        vector_nuevo: np.ndarray,
        angulo_nuevo: str   = "frontal",
        umbral:       float = UMBRAL,
) -> dict | None:
    registros = cargar_vectores_por_angulo()
    if not registros:
        return None

    por_angulo = [r for r in registros if r["angulo"] == angulo_nuevo]
    if not por_angulo:
        por_angulo = [r for r in registros if r["angulo"] == "frontal"]
    if not por_angulo:
        return None

    distancias: dict[int, dict] = {}
    for reg in por_angulo:
        uid  = reg["usuario_id"]
        dist = round(float(distancia_chi2(vector_nuevo, reg["vector"])), 4)
        if uid not in distancias or dist < distancias[uid]["distancia"]:
            distancias[uid] = {
                "usuario_id":    uid,
                "nombre":        reg["nombre"],
                "numero_cuenta": reg["numero_cuenta"],
                "rol":           reg["rol"],
                "angulo":        reg["angulo"],
                "distancia":     dist,
            }

    if not distancias:
        return None

    ordenados = sorted(distancias.values(), key=lambda x: x["distancia"])
    mejor     = ordenados[0]

    print("[RECONO] " +
          ", ".join(f"{r['nombre'].split()[0]}={r['distancia']:.4f}"
                    for r in ordenados))

    if mejor["distancia"] > RECHAZO:
        print(f"[RECONO] Desconocido dist={mejor['distancia']:.4f}")
        return None

    if len(ordenados) >= 2:
        gap = round(ordenados[1]["distancia"] - mejor["distancia"], 4)
        if gap < GAP_MIN:
            print(f"[RECONO] Ambiguo gap={gap:.4f}")
            return None

    sim_raw                = max(0.0, 1.0 - (mejor["distancia"] / MAX_DIST))
    mejor["similitud_pct"] = round(sim_raw * 100, 1)
    mejor["acceso"]        = mejor["distancia"] <= umbral

    print(f"[RECONO] → {mejor['nombre']} acceso={'SI' if mejor['acceso'] else 'NO'}")
    return mejor


# =============================================================================
#  DETECCION DE DUPLICADO FACIAL
# =============================================================================

def verificar_duplicado_facial(
        vectores_nuevos: dict,
        excluir_id:      int   = None,
        umbral:          float = UMBRAL_DUPLICADO,
) -> dict | None:
    if not vectores_nuevos:
        return None
    registros = cargar_vectores_por_angulo(excluir_id=excluir_id)
    if not registros:
        return None

    mejores: dict[int, dict] = {}
    for reg in registros:
        uid = reg["usuario_id"]
        ang = reg["angulo"]
        if ang not in vectores_nuevos:
            continue
        dist = round(float(
            distancia_chi2(vectores_nuevos[ang], reg["vector"])), 4)
        if uid not in mejores or dist < mejores[uid]["distancia"]:
            mejores[uid] = {
                "usuario_id":    uid,
                "nombre":        reg["nombre"],
                "numero_cuenta": reg["numero_cuenta"],
                "rol":           reg["rol"],
                "angulo":        ang,
                "distancia":     dist,
            }

    if not mejores:
        return None
    mejor = min(mejores.values(), key=lambda x: x["distancia"])
    print(f"[DUP] {mejor['nombre']} dist={mejor['distancia']:.4f}")
    if mejor["distancia"] <= umbral:
        print(f"[DUP] BLOQUEADO")
        return mejor
    return None


# =============================================================================
#  REGISTRO DE ACCESO
# =============================================================================

def registrar_acceso(usuario_id: int, tipo_evento: str,
                      detalle: str = None) -> bool:
    if tipo_evento not in ("entrada", "salida", "intento_fallido"):
        return False
    try:
        conn   = conectar()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT nombre, apellido_paterno, apellido_materno,
                   numero_cuenta, rol
            FROM usuarios WHERE id = ?
        """, (usuario_id,))
        row = cursor.fetchone()
        if not row:
            return False
        nombre, ap_pat, ap_mat, cuenta, rol = row
        cursor.execute("""
            INSERT INTO registro_acceso
                (usuario_id, nombre, apellido_paterno, apellido_materno,
                 numero_cuenta, rol, tipo_evento, detalle)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (usuario_id, nombre, ap_pat, ap_mat,
              cuenta, rol, tipo_evento, detalle))
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


# =============================================================================
#  CONSULTAS AUXILIARES
# =============================================================================

def tiene_vector(usuario_id: int) -> bool:
    conn = conectar()
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM vectores_por_angulo WHERE usuario_id=?",
              (usuario_id,))
    n = c.fetchone()[0]
    conn.close()
    return n > 0


def angulos_registrados(usuario_id: int) -> list:
    conn = conectar()
    c    = conn.cursor()
    c.execute("""
        SELECT angulo, n_muestras FROM vectores_por_angulo
        WHERE usuario_id = ? ORDER BY angulo
    """, (usuario_id,))
    rows = c.fetchall()
    conn.close()
    return rows