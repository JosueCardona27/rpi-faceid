"""
database.py
===========
Base de datos SQLite para el sistema de reconocimiento facial.

Caracteristicas:
  - Roles: admin, maestro, estudiante
  - Vectores LBP por angulo: frontal, perfil_der, perfil_izq (512 dims)
  - Reconocimiento por angulo consistente (frontal vs frontal)
  - Deteccion de duplicado facial en registro
  - Registro de accesos inmutable
  - Auditoria de cambios inmutable

Umbrales calibrados con datos reales:
  UMBRAL           = 0.55   distancia maxima para acceso permitido
  RECHAZO          = 0.58   por encima de esto = desconocido
  GAP_MIN          = 0.05   margen minimo entre 1° y 2° candidato
  UMBRAL_DUPLICADO = 0.20   bloquea registro si misma cara detectada

Permisos por rol:
  admin    — puede registrar admin, maestro y estudiante
  maestro  — solo puede registrar estudiantes
  estudiante — no puede registrar a nadie
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

# =============================================================================
#  CONSTANTES — ajustar aqui si cambia el hardware o iluminacion
# =============================================================================
UMBRAL           = 0.35   # dist <= UMBRAL  → acceso PERMITIDO
RECHAZO          = 0.80   # dist >  RECHAZO → DESCONOCIDO
GAP_MIN          = 0.05   # diferencia minima entre 1° y 2° candidato
MAX_DIST         = 2.75   # referencia para calcular porcentaje de similitud
UMBRAL_DUPLICADO = 0.20   # dist <= esto en registro → misma persona, bloquear

ANGULOS_VALIDOS = ("frontal", "perfil_der", "perfil_izq")
ROLES_VALIDOS   = ("admin", "maestro", "estudiante")

# Que roles puede registrar cada rol
PERMISOS_REGISTRO = {
    "admin":      ("admin", "maestro", "estudiante"),
    "maestro":    ("estudiante",),
    "estudiante": (),
}


# =============================================================================
#  CONEXION
# =============================================================================

def conectar() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode  = WAL")
    conn.execute("PRAGMA synchronous   = NORMAL")
    return conn


# =============================================================================
#  INICIALIZACION
# =============================================================================

def _crear_tablas():
    """Crea todas las tablas, indices, triggers y vistas si no existen."""
    conn = conectar()
    conn.executescript("""

        -- ── Tabla principal de usuarios ──────────────────────────────────────
        CREATE TABLE IF NOT EXISTS usuarios (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            id_usuario       TEXT    NOT NULL UNIQUE,
            nombre           TEXT    NOT NULL,
            apellido_paterno TEXT    NOT NULL,
            apellido_materno TEXT    NOT NULL DEFAULT '.',
            numero_cuenta    TEXT    UNIQUE,
            rol              TEXT    NOT NULL
                             CHECK(rol IN ('admin','maestro','estudiante')),
            registrado_por   INTEGER REFERENCES usuarios(id) ON DELETE SET NULL,
            fecha_registro   TEXT    NOT NULL
                             DEFAULT (datetime('now','localtime'))
        );

        -- ── Vectores por angulo (principal) ──────────────────────────────────
        CREATE TABLE IF NOT EXISTS vectores_por_angulo (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id    INTEGER NOT NULL
                          REFERENCES usuarios(id) ON DELETE CASCADE,
            angulo        TEXT    NOT NULL
                          CHECK(angulo IN ('frontal','perfil_der','perfil_izq')),
            vector        TEXT    NOT NULL,
            dimensiones   INTEGER NOT NULL DEFAULT 512,
            n_muestras    INTEGER NOT NULL DEFAULT 1,
            fecha_captura TEXT    NOT NULL
                          DEFAULT (datetime('now','localtime')),
            UNIQUE(usuario_id, angulo)
        );

        -- ── Historial de accesos (inmutable) ─────────────────────────────────
        CREATE TABLE IF NOT EXISTS registro_acceso (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id       INTEGER REFERENCES usuarios(id) ON DELETE SET NULL,
            nombre           TEXT    NOT NULL,
            apellido_paterno TEXT    NOT NULL,
            apellido_materno TEXT    NOT NULL,
            numero_cuenta    TEXT,
            rol              TEXT    NOT NULL
                             CHECK(rol IN ('admin','maestro','estudiante')),
            tipo_evento      TEXT    NOT NULL
                             CHECK(tipo_evento IN ('entrada','salida','intento_fallido')),
            fecha_acceso     TEXT    NOT NULL
                             DEFAULT (datetime('now','localtime')),
            detalle          TEXT
        );

        -- ── Auditoria de cambios (inmutable) ─────────────────────────────────
        CREATE TABLE IF NOT EXISTS auditoria_cambios (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            tabla            TEXT    NOT NULL,
            operacion        TEXT    NOT NULL
                             CHECK(operacion IN ('INSERT','UPDATE','DELETE')),
            registro_id      INTEGER,
            campo_modificado TEXT,
            valor_anterior   TEXT,
            valor_nuevo      TEXT,
            modificado_por   INTEGER REFERENCES usuarios(id) ON DELETE SET NULL,
            fecha            TEXT    NOT NULL
                             DEFAULT (datetime('now','localtime'))
        );

        -- ── Indices ───────────────────────────────────────────────────────────
        CREATE INDEX  IF NOT EXISTS idx_usuarios_rol      ON usuarios(rol);
        CREATE INDEX  IF NOT EXISTS idx_vpa_usuario       ON vectores_por_angulo(usuario_id);
        CREATE INDEX  IF NOT EXISTS idx_vpa_angulo        ON vectores_por_angulo(angulo);
        CREATE INDEX  IF NOT EXISTS idx_acceso_usuario    ON registro_acceso(usuario_id);
        CREATE INDEX  IF NOT EXISTS idx_acceso_fecha      ON registro_acceso(fecha_acceso);
        CREATE INDEX  IF NOT EXISTS idx_acceso_tipo       ON registro_acceso(tipo_evento);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_uq_cuenta   ON usuarios(numero_cuenta)
            WHERE numero_cuenta IS NOT NULL;

        -- ── Triggers de proteccion ────────────────────────────────────────────

        -- Bloquear numero de cuenta duplicado en INSERT
        CREATE TRIGGER IF NOT EXISTS trg_cuenta_dup_insert
        BEFORE INSERT ON usuarios
        WHEN NEW.numero_cuenta IS NOT NULL
        BEGIN
            SELECT RAISE(ABORT,'DUPLICADO: El numero de cuenta ya esta registrado.')
            WHERE EXISTS (
                SELECT 1 FROM usuarios WHERE numero_cuenta = NEW.numero_cuenta
            );
        END;

        -- Bloquear numero de cuenta duplicado en UPDATE
        CREATE TRIGGER IF NOT EXISTS trg_cuenta_dup_update
        BEFORE UPDATE OF numero_cuenta ON usuarios
        WHEN NEW.numero_cuenta IS NOT NULL
        BEGIN
            SELECT RAISE(ABORT,'DUPLICADO: El numero de cuenta pertenece a otro usuario.')
            WHERE EXISTS (
                SELECT 1 FROM usuarios
                WHERE numero_cuenta = NEW.numero_cuenta AND id != NEW.id
            );
        END;

        -- Proteger el rol — no se puede cambiar una vez asignado
        CREATE TRIGGER IF NOT EXISTS trg_proteger_rol
        BEFORE UPDATE OF rol ON usuarios
        BEGIN
            SELECT RAISE(ABORT,'PROHIBIDO: El rol de un usuario no puede modificarse.')
            WHERE OLD.rol != NEW.rol;
        END;

        -- Auditoria automatica en UPDATE de usuarios
        CREATE TRIGGER IF NOT EXISTS trg_auditoria_update
        AFTER UPDATE ON usuarios
        BEGIN
            INSERT INTO auditoria_cambios
                (tabla, operacion, registro_id, campo_modificado,
                 valor_anterior, valor_nuevo)
            SELECT 'usuarios','UPDATE',NEW.id,'nombre',OLD.nombre,NEW.nombre
            WHERE OLD.nombre != NEW.nombre;

            INSERT INTO auditoria_cambios
                (tabla, operacion, registro_id, campo_modificado,
                 valor_anterior, valor_nuevo)
            SELECT 'usuarios','UPDATE',NEW.id,'apellido_paterno',
                   OLD.apellido_paterno,NEW.apellido_paterno
            WHERE OLD.apellido_paterno != NEW.apellido_paterno;

            INSERT INTO auditoria_cambios
                (tabla, operacion, registro_id, campo_modificado,
                 valor_anterior, valor_nuevo)
            SELECT 'usuarios','UPDATE',NEW.id,'numero_cuenta',
                   OLD.numero_cuenta,NEW.numero_cuenta
            WHERE OLD.numero_cuenta IS NOT NEW.numero_cuenta;

        END;

        -- Auditoria automatica en DELETE de usuarios
        CREATE TRIGGER IF NOT EXISTS trg_auditoria_delete
        AFTER DELETE ON usuarios
        BEGIN
            INSERT INTO auditoria_cambios
                (tabla, operacion, registro_id, campo_modificado,
                 valor_anterior, valor_nuevo)
            VALUES (
                'usuarios','DELETE',OLD.id,'ELIMINADO',
                OLD.nombre||' '||OLD.apellido_paterno||' ('||OLD.rol||')',
                NULL
            );
        END;

        -- Proteger registro_acceso — inmutable
        CREATE TRIGGER IF NOT EXISTS trg_proteger_acceso_delete
        BEFORE DELETE ON registro_acceso
        BEGIN
            SELECT RAISE(ABORT,'PROHIBIDO: Los registros de acceso son inmutables.');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_proteger_acceso_update
        BEFORE UPDATE ON registro_acceso
        BEGIN
            SELECT RAISE(ABORT,'PROHIBIDO: Los registros de acceso no pueden modificarse.');
        END;

        -- Proteger auditoria_cambios — inmutable
        CREATE TRIGGER IF NOT EXISTS trg_proteger_auditoria_delete
        BEFORE DELETE ON auditoria_cambios
        BEGIN
            SELECT RAISE(ABORT,'PROHIBIDO: Los registros de auditoria son inmutables.');
        END;

        CREATE TRIGGER IF NOT EXISTS trg_proteger_auditoria_update
        BEFORE UPDATE ON auditoria_cambios
        BEGIN
            SELECT RAISE(ABORT,'PROHIBIDO: Los registros de auditoria no pueden modificarse.');
        END;

        -- ── Vistas ────────────────────────────────────────────────────────────

        CREATE VIEW IF NOT EXISTS vista_usuarios AS
        SELECT
            u.id,
            u.nombre,
            u.apellido_paterno,
            u.apellido_materno,
            u.numero_cuenta,
            u.rol,
            u.registrado_por,
            u.fecha_registro,
            COUNT(va.id) AS angulos_registrados
        FROM usuarios u
        LEFT JOIN vectores_por_angulo va ON u.id = va.usuario_id
        GROUP BY u.id;

        CREATE VIEW IF NOT EXISTS vista_accesos_recientes AS
        SELECT * FROM registro_acceso
        WHERE fecha_acceso >= datetime('now','-30 days','localtime')
        ORDER BY fecha_acceso DESC;

        CREATE VIEW IF NOT EXISTS vista_intentos_fallidos AS
        SELECT * FROM registro_acceso
        WHERE tipo_evento = 'intento_fallido'
        ORDER BY fecha_acceso DESC;

        CREATE VIEW IF NOT EXISTS vista_sin_vectores AS
        SELECT u.id, u.nombre, u.apellido_paterno, u.rol, u.fecha_registro
        FROM usuarios u
        LEFT JOIN vectores_por_angulo va ON u.id = va.usuario_id
        WHERE va.id IS NULL;

    """)
    conn.commit()
    conn.close()

_crear_tablas()


# =============================================================================
#  PERMISOS
# =============================================================================

def puede_registrar(rol_registrador: str, rol_nuevo: str) -> bool:
    """
    Valida si un rol tiene permiso para registrar otro rol.
      admin   → puede registrar admin, maestro, estudiante
      maestro → solo puede registrar estudiante
      estudiante → no puede registrar a nadie
    """
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
        registrado_por:   int = None,
        rol_registrador:  str = None,
        **kwargs,
) -> int:
    """
    Registra un nuevo usuario.

    rol_registrador: rol de quien hace el registro (para validar permisos).
                     Si es None, se omite la validacion de permisos.
    Retorna el id del nuevo usuario, o -1 si hubo error.
    """
    if rol not in ROLES_VALIDOS:
        print(f"[ERROR] Rol '{rol}' no valido.")
        return -1

    if rol_registrador is not None:
        if not puede_registrar(rol_registrador, rol):
            print(f"[ERROR] '{rol_registrador}' no tiene permiso para registrar '{rol}'.")
            return -1

    ap_mat = apellido_materno.strip() if apellido_materno.strip() else "."

    try:
        conn   = conectar()
        cursor = conn.cursor()
        import time as _time
        sufijo     = numero_cuenta if numero_cuenta else str(int(_time.time() * 1000))[-8:]
        id_usuario = f"{rol[:3].upper()}-{sufijo}"

        cursor.execute("""
            INSERT INTO usuarios
                (id_usuario, nombre, apellido_paterno, apellido_materno,
                 numero_cuenta, rol, registrado_por)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (id_usuario, nombre, apellido_paterno, ap_mat,
              numero_cuenta or None,
              rol, registrado_por))
        conn.commit()
        uid = cursor.lastrowid
        print(f"[DB] Usuario registrado ID={uid}: "
              f"{nombre} {apellido_paterno} ({rol})")
        return uid
    except sqlite3.IntegrityError as e:
        print(f"[DB] Error de integridad: {e}")
        return -1
    finally:
        conn.close()


def eliminar_usuario(usuario_id: int) -> bool:
    """
    Elimina un usuario y sus vectores (cascada).
    El registro de auditoria queda intacto.
    La persona puede volver a registrarse despues de ser eliminada.
    """
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
    """Lista usuarios desde vista_usuarios. Incluye conteo de angulos."""
    conn   = conectar()
    cursor = conn.cursor()
    if rol:
        cursor.execute("""
            SELECT id, nombre, apellido_paterno, apellido_materno,
                   numero_cuenta, rol, angulos_registrados
            FROM vista_usuarios
            WHERE rol = ?
            ORDER BY apellido_paterno, nombre
        """, (rol,))
    else:
        cursor.execute("""
            SELECT id, nombre, apellido_paterno, apellido_materno,
                   numero_cuenta, rol, angulos_registrados
            FROM vista_usuarios
            ORDER BY apellido_paterno, nombre
        """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def obtener_usuario(usuario_id: int) -> dict | None:
    """Retorna un dict con los datos del usuario, o None si no existe."""
    conn   = conectar()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, nombre, apellido_paterno, apellido_materno,
               numero_cuenta, rol, registrado_por, fecha_registro
        FROM usuarios WHERE id = ?
    """, (usuario_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    keys = ("id","nombre","apellido_paterno","apellido_materno",
            "numero_cuenta","rol","registrado_por","fecha_registro")
    return dict(zip(keys, row))


# =============================================================================
#  GUARDAR VECTORES
# =============================================================================

def guardar_vectores_por_angulo(usuario_id: int,
                                 vectores_por_paso: dict) -> int:
    """
    Guarda vectores promediados por angulo.

    vectores_por_paso formato:
      {
        "frontal":    {"vectores": [array(512), ...]},
        "perfil_der": {"vectores": [array(512), ...]},
        "perfil_izq": {"vectores": [array(512), ...]},
      }

    Retorna el numero de angulos guardados exitosamente.
    """
    conn   = conectar()
    cursor = conn.cursor()
    total  = 0

    for angulo, datos in vectores_por_paso.items():
        if angulo not in ANGULOS_VALIDOS:
            print(f"[DB] Angulo '{angulo}' no valido. Omitido.")
            continue

        vecs = datos.get("vectores", [])
        if not vecs:
            print(f"[DB] Angulo '{angulo}' sin muestras.")
            continue

        v_prom = np.mean(vecs, axis=0).astype(np.float32)

        if len(v_prom) != VECTOR_DIM:
            print(f"[DB] '{angulo}': {len(v_prom)} dims != {VECTOR_DIM}. Omitido.")
            continue

        cursor.execute("""
            INSERT OR REPLACE INTO vectores_por_angulo
                (usuario_id, angulo, vector, n_muestras)
            VALUES (?, ?, ?, ?)
        """, (usuario_id, angulo,
              json.dumps(v_prom.tolist()),
              len(vecs)))

        print(f"[DB] ID={usuario_id} angulo='{angulo}' muestras={len(vecs)}")
        total += 1

    conn.commit()
    conn.close()
    print(f"[DB] {total} angulos guardados para usuario ID={usuario_id}.")
    return total


def guardar_vector_unico(usuario_id: int, vectores: list,
                          pesos_lista=None):
    """Compatibilidad — guarda vector frontal en vectores_por_angulo."""
    if not vectores:
        return
    v_prom = np.mean(vectores, axis=0).astype(np.float32)
    guardar_vectores_por_angulo(
        usuario_id,
        {"frontal": {"vectores": [v_prom]}}
    )


# =============================================================================
#  CARGAR VECTORES
# =============================================================================

def _tablas_existentes(conn) -> set:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cur.fetchall()}


def cargar_vectores_por_angulo(excluir_id: int = None) -> list:
    """
    Carga todos los vectores activos de la BD.

    excluir_id: omite ese usuario_id (para no comparar el registro
                temporal recien creado contra si mismo).

    Los usuarios eliminados no aparecen porque sus vectores se borran
    en cascada, lo que permite re-registro tras eliminacion.
    La auditoria no interfiere con esto.
    """
    conn      = conectar()
    cursor    = conn.cursor()
    resultado = []
    tablas    = _tablas_existentes(conn)

    if "vectores_por_angulo" in tablas:
        cursor.execute("""
            SELECT
                u.id,
                u.nombre || ' ' || u.apellido_paterno || ' ' || u.apellido_materno,
                u.numero_cuenta,
                u.rol,
                va.angulo,
                va.vector,
                va.n_muestras
            FROM vectores_por_angulo va
            JOIN usuarios u ON u.id = va.usuario_id
            ORDER BY u.id, va.angulo
        """)
        for uid, nombre, cuenta, rol, angulo, vjson, n in cursor.fetchall():
            if excluir_id is not None and uid == excluir_id:
                continue
            v = np.array(json.loads(vjson), dtype=np.float32)
            if len(v) != VECTOR_DIM:
                continue
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
    """
    Identifica a quien pertenece vector_nuevo comparando SOLO contra
    vectores del mismo angulo almacenado (frontal vs frontal, etc.)

    Logica de decision:
      dist > RECHAZO          → None (desconocido)
      gap 1°-2° < GAP_MIN     → None (ambiguo)
      dist <= UMBRAL           → acceso=True
      UMBRAL < dist <= RECHAZO → acceso=False (conocido pero denegado)

    Siempre imprime candidatos en terminal para facilitar calibracion.
    """
    registros = cargar_vectores_por_angulo()
    if not registros:
        return None

    # Calcular la distancia minima por USUARIO (usando todos sus angulos)
    # Asi un usuario con 3 angulos sigue siendo UN candidato, no tres
    distancias: dict[int, dict] = {}
    for reg in registros:
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

    # Log siempre para calibracion
    print("[RECONO] Candidatos: " +
          ", ".join(f"{r['nombre'].split()[0]}={r['distancia']:.4f}"
                    for r in ordenados))

    # Rechazo absoluto
    if mejor["distancia"] > RECHAZO:
        print(f"[RECONO] Desconocido — "
              f"dist={mejor['distancia']:.4f} > RECHAZO={RECHAZO}")
        return None

    # GAP solo aplica cuando hay mas de un USUARIO diferente
    # (no entre angulos del mismo usuario)
    if len(ordenados) >= 2:
        gap = round(ordenados[1]["distancia"] - mejor["distancia"], 4)
        if gap < GAP_MIN:
            print(f"[RECONO] Ambiguo — "
                  f"gap={gap:.4f} < GAP_MIN={GAP_MIN}")
            return None

    sim_raw                = max(0.0, 1.0 - (mejor["distancia"] / MAX_DIST))
    mejor["similitud_pct"] = round(sim_raw * 100, 1)
    mejor["acceso"]        = mejor["distancia"] <= umbral

    print(f"[RECONO] → {mejor['nombre']} | "
          f"dist={mejor['distancia']:.4f} | "
          f"acceso={'SI' if mejor['acceso'] else 'NO'}")
    return mejor


# =============================================================================
#  DETECCION DE DUPLICADO FACIAL
# =============================================================================

def verificar_duplicado_facial(
        vectores_nuevos: dict,
        excluir_id:      int   = None,
        umbral:          float = UMBRAL_DUPLICADO,
) -> dict | None:
    """
    Verifica si la cara que se intenta registrar ya existe en la BD.

    Compara SOLO angulos que ya fueron capturados (puede llamarse
    con 1, 2 o 3 angulos durante el registro progresivo).

    dist <= umbral → misma persona → bloquear registro
    dist >  umbral → persona diferente → permitir registro

    Los usuarios eliminados no bloquean el re-registro porque
    sus vectores se borran en cascada.
    """
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
        dist = round(float(distancia_chi2(vectores_nuevos[ang],
                                           reg["vector"])), 4)
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
    print(f"[DUP] Mejor match: {mejor['nombre']} "
          f"dist={mejor['distancia']:.4f} umbral={umbral}")

    if mejor["distancia"] <= umbral:
        print(f"[DUP] BLOQUEADO — misma cara detectada.")
        return mejor

    return None


# =============================================================================
#  REGISTRO DE ACCESO
# =============================================================================

def registrar_acceso(
        usuario_id:  int,
        tipo_evento: str,
        detalle:     str = None,
) -> bool:
    """
    Registra un evento de entrada / salida / intento_fallido.
    Copia los datos del usuario en el momento del evento (inmutable).
    """
    if tipo_evento not in ("entrada", "salida", "intento_fallido"):
        print(f"[DB] tipo_evento '{tipo_evento}' no valido.")
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
            print(f"[DB] Usuario ID={usuario_id} no encontrado.")
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
        print(f"[DB] Acceso: ID={usuario_id} evento={tipo_evento}")
        return True
    except sqlite3.Error as e:
        print(f"[DB] Error al registrar acceso: {e}")
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
    """Retorna lista de (angulo, n_muestras) para el usuario."""
    conn = conectar()
    c    = conn.cursor()
    c.execute("""
        SELECT angulo, n_muestras
        FROM vectores_por_angulo
        WHERE usuario_id = ?
        ORDER BY angulo
    """, (usuario_id,))
    rows = c.fetchall()
    conn.close()
    return rows


def accesos_recientes(usuario_id: int = None, limite: int = 50) -> list:
    """Retorna los ultimos accesos, opcionalmente filtrados por usuario."""
    conn   = conectar()
    cursor = conn.cursor()
    if usuario_id:
        cursor.execute("""
            SELECT * FROM vista_accesos_recientes
            WHERE usuario_id = ?
            LIMIT ?
        """, (usuario_id, limite))
    else:
        cursor.execute("""
            SELECT * FROM vista_accesos_recientes
            LIMIT ?
        """, (limite,))
    rows = cursor.fetchall()
    conn.close()
    return rows