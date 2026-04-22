"""
api.py  ─  Servidor REST  (corre en la Raspberry Pi)
====================================================
Expone TODOS los datos que el dashboard necesita:
  · KPIs y accesos recientes  →  ResumenView
  · Historial completo         →  AccesosView
  · Estadísticas               →  StatsView
  · Usuarios CRUD              →  GestionView
  · Vectores biométricos       →  registro_facial_dash
  · Perfil del usuario         →  PerfilView

Arranque rápido (desarrollo):
    cd /home/pi/labcontrol
    source venv/bin/activate
    python src/api.py

Arranque automático (producción, ver setup_hotspot.sh):
    sudo systemctl start labcontrol-api

Variables de entorno:
    LABCONTROL_TOKEN   token de autenticación (default: labcontrol-secret-2025)
    LABCONTROL_PORT    puerto de escucha      (default: 5000)
"""

import os
import sys
import json
import sqlite3
import numpy as np
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify

# Ruta a database.py 
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from database import (
    DB_PATH,
    conectar,
    hash_contrasena,
    registrar_usuario,
    eliminar_usuario,
    obtener_usuario,
    listar_usuarios,
    guardar_vectores_por_angulo,
    cargar_vectores_por_angulo,
    registrar_acceso,
    tiene_vector,
    angulos_registrados,
    ROLES_VALIDOS,
    ANGULOS_VALIDOS,
    UMBRAL,
    RECHAZO,
)

# Config
TOKEN = os.environ.get("LABCONTROL_TOKEN", "labcontrol-secret-2025")
PORT  = int(os.environ.get("LABCONTROL_PORT", 5000))
HOST  = "0.0.0.0"

app = Flask(__name__)


#  HELPERS 

def _requiere_token(f):
    @wraps(f)
    def _inner(*args, **kwargs):
        if request.headers.get("X-LabControl-Token", "") != TOKEN:
            return jsonify({"ok": False, "error": "Token inválido."}), 401
        return f(*args, **kwargs)
    return _inner


def _ok(data=None, **extra):
    r = {"ok": True}
    if data is not None:
        r["data"] = data
    r.update(extra)
    return jsonify(r), 200


def _err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


def _row_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def _conn() -> sqlite3.Connection:
    """Conexión con row_factory activado."""
    c = conectar()
    c.row_factory = sqlite3.Row
    return c

#  PING  (sin token — para descubrimiento)

@app.route("/ping")
def ping():
    return jsonify({
        "ok":      True,
        "server":  "LabControl API",
        "version": "2.0",
        "time":    datetime.now().isoformat(),
    })


#  RESUMEN  —  KPIs + últimos accesos  (ResumenView)

@app.route("/resumen")
@_requiere_token
def resumen():
    """
    Devuelve los 3 KPIs y los últimos N accesos en una sola petición,
    para que ResumenView no tenga que hacer múltiples llamadas.
    """
    n = int(request.args.get("n", 4))
    try:
        conn = _conn()
        c    = conn.cursor()

        # KPI: personas dentro ahora
        c.execute("""
            SELECT COUNT(*) FROM (
                SELECT usuario_id FROM registro_acceso
                WHERE id IN (
                    SELECT MAX(id) FROM registro_acceso
                    WHERE DATE(fecha_acceso, 'localtime') = DATE('now','localtime')
                    GROUP BY usuario_id
                ) AND tipo_evento = 'entrada'
            )
        """)
        dentro = c.fetchone()[0]

        # KPI: entradas hoy
        c.execute("""
            SELECT COUNT(*) FROM registro_acceso
            WHERE tipo_evento = 'entrada'
              AND DATE(fecha_acceso,'localtime') = DATE('now','localtime')
        """)
        hoy = c.fetchone()[0]

        # KPI: entradas esta semana
        c.execute("""
            SELECT COUNT(*) FROM registro_acceso
            WHERE tipo_evento = 'entrada'
              AND DATE(fecha_acceso,'localtime') >= DATE('now','-6 days','localtime')
        """)
        semana = c.fetchone()[0]

        # Últimos N accesos
        c.execute("""
            SELECT nombre || ' ' || apellido_paterno,
                   numero_cuenta, rol, tipo_evento, fecha_acceso
            FROM registro_acceso
            ORDER BY id DESC LIMIT ?
        """, (n,))
        ultimos = [
            {"nombre":  (r[0] or "").strip(),
             "cuenta":  r[1] or "—",
             "rol":     r[2] or "—",
             "tipo":    r[3],
             "hora":    (r[4] or "")[11:16]}
            for r in c.fetchall()
        ]

        conn.close()
        return _ok({
            "kpi_dentro": dentro,
            "kpi_hoy":    hoy,
            "kpi_semana": semana,
            "ultimos":    ultimos,
        })
    except Exception as e:
        return _err(str(e), 500)


#  ACCESOS  —  historial completo

@app.route("/accesos")
@_requiere_token
def accesos():
    """
    GET /accesos?rol=estudiante&limit=500
    Historial completo (o filtrado por rol).
    """
    rol_filtro = request.args.get("rol", "")
    limit      = min(int(request.args.get("limit", 500)), 2000)
    try:
        conn = _conn()
        c    = conn.cursor()
        sql  = """
            SELECT ra.nombre || ' ' || ra.apellido_paterno,
                   ra.numero_cuenta, ra.rol,
                   ra.tipo_evento,   ra.fecha_acceso,
                   ed.grado,         ed.grupo
            FROM registro_acceso ra
            LEFT JOIN estudiantes_detalle ed ON ed.usuario_id = ra.usuario_id
        """
        params: tuple = ()
        if rol_filtro:
            sql    += " WHERE ra.rol = ?"
            params  = (rol_filtro,)
        sql += " ORDER BY ra.id ASC LIMIT ?"
        params = params + (limit,)
        c.execute(sql, params)
        rows = c.fetchall()
        conn.close()

        result = []
        for i, r in enumerate(rows, 1):
            nombre, cuenta, rol, tipo, fa, grado, grupo = r
            es_staff = rol in ("maestro", "admin")
            result.append({
                "num":    i,
                "nombre": (nombre or "").strip(),
                "cuenta": cuenta or "—",
                "grado":  "—" if es_staff else (str(grado) if grado else "—"),
                "grupo":  "—" if es_staff else (grupo or "—"),
                "rol":    rol or "—",
                "tipo":   tipo,
                "fecha":  (fa or "")[:10]   or "—",
                "hora":   (fa or "")[11:16] or "—",
            })
        return _ok(result)
    except Exception as e:
        return _err(str(e), 500)


#  Estadísticas  (StatsView)

@app.route("/stats")
@_requiere_token
def stats():
    """
    Devuelve top7, distribución por rol y accesos por hora en una
    sola petición para StatsView.
    """
    try:
        conn = _conn()
        c    = conn.cursor()

        # Top 7 visitantes
        c.execute("""
            SELECT ra.nombre || ' ' || ra.apellido_paterno,
                   ra.numero_cuenta, ra.rol,
                   COUNT(*) AS visitas,
                   ed.grado, ed.grupo
            FROM registro_acceso ra
            LEFT JOIN estudiantes_detalle ed ON ed.usuario_id = ra.usuario_id
            WHERE ra.tipo_evento = 'entrada'
            GROUP BY ra.usuario_id
            ORDER BY visitas DESC LIMIT 7
        """)
        top7 = []
        for nom, cuenta, rol, vis, grado, grupo in c.fetchall():
            nom = (nom or "").strip()
            parts = nom.split(" ", 1)
            if rol in ("maestro", "admin"):
                sub = rol.capitalize()
            else:
                sub = (f"{grado}° {grupo}" if grado and grupo else
                       f"{grado}°"          if grado           else "—")
            ini1 = parts[0][0].upper() if parts[0] else "?"
            ini2 = parts[1][0].upper() if len(parts) > 1 and parts[1] else "?"
            top7.append({
                "nombre":  nom,
                "cuenta":  cuenta or "—",
                "rol":     rol or "—",
                "visitas": vis,
                "sub":     sub,
                "ini":     ini1 + ini2,
            })

        # Accesos por rol
        c.execute("""
            SELECT rol, COUNT(*) FROM registro_acceso
            WHERE tipo_evento = 'entrada'
            GROUP BY rol ORDER BY COUNT(*) DESC
        """)
        por_rol = [{"rol": r[0], "cnt": r[1]} for r in c.fetchall()]

        # Accesos por hora (últimos 7 días)
        c.execute("""
            SELECT CAST(STRFTIME('%H', fecha_acceso,'localtime') AS INTEGER) AS hr,
                   COUNT(*) AS cnt
            FROM registro_acceso
            WHERE tipo_evento = 'entrada'
              AND DATE(fecha_acceso,'localtime') >= DATE('now','-6 days','localtime')
            GROUP BY hr ORDER BY hr
        """)
        por_hora = [{"hora": r[0], "cnt": r[1]} for r in c.fetchall()]

        conn.close()
        return _ok({
            "top7":     top7,
            "por_rol":  por_rol,
            "por_hora": por_hora,
        })
    except Exception as e:
        return _err(str(e), 500)


#  PERFIL STATS

@app.route("/perfil/stats")
@_requiere_token
def perfil_stats():
    """Accesos y usuarios registrados en el mes actual."""
    try:
        conn = _conn()
        c    = conn.cursor()
        c.execute("""
            SELECT COUNT(*) FROM registro_acceso
            WHERE tipo_evento = 'entrada'
              AND DATE(fecha_acceso,'localtime') >=
                  DATE('now','start of month','localtime')
        """)
        accesos = c.fetchone()[0]
        c.execute("""
            SELECT COUNT(*) FROM usuarios
            WHERE DATE(fecha_registro,'localtime') >=
                  DATE('now','start of month','localtime')
        """)
        reg = c.fetchone()[0]
        conn.close()
        return _ok({"accesos_mes": accesos, "usuarios_mes": reg})
    except Exception as e:
        return _err(str(e), 500)


#  USUARIOS  —  CRUD  (GestionView + login)

@app.route("/usuarios", methods=["GET"])
@_requiere_token
def usuarios_listar():
    """GET /usuarios?rol=estudiante"""
    rol  = request.args.get("rol")
    rows = listar_usuarios(rol)
    keys = ("id", "nombre", "apellido_paterno", "apellido_materno",
            "numero_cuenta", "correo", "rol", "grado", "grupo",
            "angulos_registrados")
    return _ok([dict(zip(keys, r)) for r in rows])


@app.route("/usuarios/<int:uid>", methods=["GET"])
@_requiere_token
def usuario_obtener(uid):
    u = obtener_usuario(uid)
    return _ok(u) if u else _err("No encontrado.", 404)


@app.route("/usuarios", methods=["POST"])
@_requiere_token
def usuario_crear():
    """
    POST /usuarios
    {
        "nombre", "apellido_paterno", "apellido_materno",
        "rol", "numero_cuenta",
        "correo", "contrasena",   ← maestro/admin
        "grado", "grupo",         ← estudiante
        "registrado_por"
    }
    """
    d   = request.get_json(silent=True) or {}
    uid = registrar_usuario(
        nombre           = d.get("nombre", "").strip(),
        apellido_paterno = d.get("apellido_paterno", "").strip(),
        apellido_materno = d.get("apellido_materno", ""),
        rol              = d.get("rol", "estudiante"),
        numero_cuenta    = d.get("numero_cuenta"),
        correo           = d.get("correo"),
        contrasena       = d.get("contrasena"),
        grado            = d.get("grado"),
        grupo            = d.get("grupo"),
        registrado_por   = d.get("registrado_por"),
    )
    if uid == -1:
        return _err("No se pudo registrar. Verifica los datos.", 422)
    return jsonify({"ok": True, "data": {"id": uid},
                    "message": "Usuario creado."}), 201


@app.route("/usuarios/<int:uid>", methods=["PUT"])
@_requiere_token
def usuario_editar(uid):
    """PUT /usuarios/{id}  — actualiza campos presentes en el body."""
    d = request.get_json(silent=True) or {}
    if not d:
        return _err("Body vacío.")
    try:
        conn   = _conn()
        cur    = conn.cursor()
        sets, params = [], []
        for f in ("nombre", "apellido_paterno", "apellido_materno",
                  "numero_cuenta", "correo"):
            if f in d:
                sets.append(f"{f}=?")
                params.append(d[f])
        if d.get("contrasena"):
            sets.append("contrasena=?")
            params.append(hash_contrasena(d["contrasena"]))
        if not sets:
            conn.close()
            return _err("Nada que actualizar.")
        sets.append("fecha_actualizacion=?")
        params.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if "actualizado_por" in d:
            sets.append("actualizado_por=?")
            params.append(d["actualizado_por"])
        params.append(uid)
        cur.execute(f"UPDATE usuarios SET {', '.join(sets)} WHERE id=?",
                    params)
        # estudiantes_detalle
        if "grado" in d or "grupo" in d:
            cur.execute("SELECT id FROM estudiantes_detalle WHERE usuario_id=?",
                        (uid,))
            if cur.fetchone():
                cur.execute("UPDATE estudiantes_detalle SET grado=?,grupo=? "
                            "WHERE usuario_id=?",
                            (d.get("grado"), d.get("grupo",""), uid))
            else:
                cur.execute("INSERT INTO estudiantes_detalle "
                            "(usuario_id,grado,grupo) VALUES (?,?,?)",
                            (uid, d.get("grado"), d.get("grupo","")))
        conn.commit()
        conn.close()
        return _ok(message="Actualizado.")
    except Exception as e:
        return _err(str(e), 500)


@app.route("/usuarios/<int:uid>", methods=["DELETE"])
@_requiere_token
def usuario_borrar(uid):
    ok = eliminar_usuario(uid)
    return _ok(message="Eliminado.") if ok else _err("No se pudo eliminar.", 500)


@app.route("/usuarios/cuenta/<string:cuenta>", methods=["DELETE"])
@_requiere_token
def usuario_borrar_por_cuenta(cuenta):
    """Eliminar por número de cuenta (útil al sincronizar bajas)."""
    try:
        conn = _conn()
        cur  = conn.cursor()
        cur.execute("SELECT id FROM usuarios WHERE numero_cuenta=?", (cuenta,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return _err("No encontrado.", 404)
        ok = eliminar_usuario(row["id"])
        return _ok(message="Eliminado.") if ok else _err("Error al eliminar.", 500)
    except Exception as e:
        return _err(str(e), 500)


# Login

@app.route("/login", methods=["POST"])
@_requiere_token
def login():
    """
    POST /login
    { "usuario": "cuenta_o_correo", "contrasena": "texto_plano" }
    Devuelve los datos del usuario si las credenciales son correctas.
    """
    import hashlib
    d        = request.get_json(silent=True) or {}
    usuario  = (d.get("usuario") or "").strip()
    pwd      = d.get("contrasena", "")
    if not usuario or not pwd:
        return _err("Faltan credenciales.", 400)

    try:
        conn = _conn()
        cur  = conn.cursor()
        if "@" in usuario:
            cur.execute("""
                SELECT id, nombre, apellido_paterno, apellido_materno,
                       numero_cuenta, correo, rol, contrasena
                FROM usuarios
                WHERE LOWER(correo) = LOWER(?) AND rol IN ('admin','maestro')
            """, (usuario,))
        else:
            cur.execute("""
                SELECT id, nombre, apellido_paterno, apellido_materno,
                       numero_cuenta, correo, rol, contrasena
                FROM usuarios
                WHERE numero_cuenta = ? AND rol IN ('admin','maestro')
            """, (usuario,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return _err("Usuario no encontrado o sin permisos.", 404)
        pwd_hash = hashlib.sha256(pwd.encode()).hexdigest()
        if pwd_hash != row["contrasena"]:
            return _err("Contraseña incorrecta.", 401)

        return _ok({
            "id":               row["id"],
            "nombre":           row["nombre"],
            "apellido_paterno": row["apellido_paterno"],
            "apellido_materno": row["apellido_materno"],
            "numero_cuenta":    row["numero_cuenta"],
            "correo":           row["correo"],
            "rol":              row["rol"],
        })
    except Exception as e:
        return _err(str(e), 500)

#  BIOMETRÍA

@app.route("/usuarios/<int:uid>/vectores", methods=["POST"])
@_requiere_token
def vectores_guardar(uid):
    """
    POST /usuarios/{id}/vectores
    {
        "vectores_angulo": {
            "frontal":    { "vectores": [[...512 floats...], ...] },
            "perfil_der": { "vectores": [...] },
            "perfil_izq": { "vectores": [...] }
        }
    }
    """
    d   = request.get_json(silent=True) or {}
    raw = d.get("vectores_angulo", {})
    if not raw:
        return _err("No se recibieron vectores.")

    vectores_angulo = {}
    for angulo, datos in raw.items():
        if angulo not in ANGULOS_VALIDOS:
            continue
        vecs = datos.get("vectores", [])
        if not vecs:
            continue
        vectores_angulo[angulo] = {
            "vectores": [np.array(v, dtype=np.float32) for v in vecs]
        }
    if not vectores_angulo:
        return _err("Ningún ángulo válido.")

    total = guardar_vectores_por_angulo(uid, vectores_angulo)
    return _ok({"angulos_guardados": total})


@app.route("/usuarios/<int:uid>/vectores", methods=["GET"])
@_requiere_token
def vectores_info(uid):
    """Estado de la biometría de un usuario."""
    return _ok({
        "tiene_vector": tiene_vector(uid),
        "angulos":      [r[0] for r in angulos_registrados(uid)],
    })


#  SINCRONIZACIÓN COMPLETA  (push desde laptop)

@app.route("/sync/push", methods=["POST"])
@_requiere_token
def sync_push():
    """
    Recibe un usuario completo + vectores desde la laptop.
    Crea el usuario si no existe, lo actualiza si ya existe
    (identificado por numero_cuenta).
    """
    d        = request.get_json(silent=True) or {}
    u_data   = d.get("usuario", {})
    vec_data = d.get("vectores_angulo", {})

    if not u_data:
        return _err("Faltan datos del usuario.")

    cuenta = u_data.get("numero_cuenta", "")

    # ¿Ya existe?
    try:
        conn = _conn()
        cur  = conn.cursor()
        cur.execute("SELECT id FROM usuarios WHERE numero_cuenta=?", (cuenta,))
        row = cur.fetchone()
        conn.close()
        uid_ex = row["id"] if row else None
    except Exception as e:
        return _err(f"Error BD: {e}", 500)

    creado = False
    if uid_ex:
        # Actualizar
        sets, params = [], []
        for f in ("nombre", "apellido_paterno", "apellido_materno", "correo"):
            if f in u_data:
                sets.append(f"{f}=?"); params.append(u_data[f])
        if u_data.get("contrasena_hash"):        # ya viene hasheado
            sets.append("contrasena=?")
            params.append(u_data["contrasena_hash"])
        if sets:
            params.append(uid_ex)
            try:
                conn = _conn()
                conn.execute(
                    f"UPDATE usuarios SET {', '.join(sets)} WHERE id=?",
                    params)
                conn.commit(); conn.close()
            except Exception as e:
                return _err(f"Error actualizando: {e}", 500)
        uid = uid_ex
    else:
        uid = registrar_usuario(
            nombre           = u_data.get("nombre","").strip(),
            apellido_paterno = u_data.get("apellido_paterno","").strip(),
            apellido_materno = u_data.get("apellido_materno",""),
            rol              = u_data.get("rol","estudiante"),
            numero_cuenta    = cuenta,
            correo           = u_data.get("correo"),
            contrasena       = u_data.get("contrasena"),
            grado            = u_data.get("grado"),
            grupo            = u_data.get("grupo"),
            registrado_por   = u_data.get("registrado_por"),
        )
        if uid == -1:
            return _err("No se pudo crear el usuario.", 422)
        creado = True

    # Guardar vectores
    angulos_ok = 0
    if vec_data:
        vang = {}
        for ang, datos in vec_data.items():
            if ang not in ANGULOS_VALIDOS: continue
            vecs = datos.get("vectores", [])
            if not vecs: continue
            vang[ang] = {"vectores": [np.array(v, dtype=np.float32)
                                       for v in vecs]}
        if vang:
            angulos_ok = guardar_vectores_por_angulo(uid, vang)

    code = 201 if creado else 200
    return jsonify({
        "ok": True,
        "data": {
            "uid":    uid,
            "creado": creado,
            "angulos_sincronizados": angulos_ok,
        },
        "message": "Sincronización exitosa.",
    }), code


@app.route("/sync/export")
@_requiere_token
def sync_export():
    """Exporta toda la BD como JSON (backup / carga inicial en laptop)."""
    try:
        conn = _conn()
        cur  = conn.cursor()

        cur.execute("""
            SELECT u.id, u.nombre, u.apellido_paterno, u.apellido_materno,
                   u.numero_cuenta, u.correo, u.contrasena, u.rol,
                   u.fecha_registro, u.registrado_por,
                   ed.grado, ed.grupo
            FROM usuarios u
            LEFT JOIN estudiantes_detalle ed ON ed.usuario_id = u.id
        """)
        usuarios = [_row_dict(r) for r in cur.fetchall()]

        cur.execute(
            "SELECT usuario_id, angulo, vector, dimensiones, n_muestras "
            "FROM vectores_por_angulo")
        vectores = [_row_dict(r) for r in cur.fetchall()]

        conn.close()
        return _ok({
            "timestamp": datetime.now().isoformat(),
            "usuarios":  usuarios,
            "vectores":  vectores,
        })
    except Exception as e:
        return _err(str(e), 500)

if __name__ == "__main__":
    print("=" * 56)
    print("  LabControl API  v2.0  —  corriendo en la Raspberry Pi")
    print(f"  URL   → http://0.0.0.0:{PORT}")
    print(f"  Token → {TOKEN[:6]}{'*' * max(0, len(TOKEN)-6)}")
    print("=" * 56)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)