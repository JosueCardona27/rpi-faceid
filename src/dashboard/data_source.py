"""
data_source.py  ─  Capa de datos unificada  (laptop)
=====================================================
Expone las MISMAS funciones que el bloque de datos original de
dashboard_views.py, pero con auto-detección de fuente:

  · Si la Raspberry Pi está accesible via WiFi  → usa api_client.py
  · Si no hay conexión                          → usa la BD local (SQLite)

De esta forma dashboard_views.py solo necesita cambiar sus imports:

    # ANTES (local):
    from dashboard_views import _kpi_dentro, _kpi_hoy, ...

    # DESPUÉS (auto):
    from data_source import (
        kpi_dentro, kpi_hoy, kpi_semana,
        ultimos_accesos, accesos_todos,
        top7, stats_rol, stats_hora,
        perfil_stats,
    )

El resto de dashboard_views.py no cambia.
"""

import os
import sys
from typing import Optional

# ── Capa local (fallback) ─────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def _db():
    from database import conectar
    return conectar()


# ── Cliente API (opcional) ────────────────────────────────────────────
try:
    from api_client import get_client, ClienteNoDisponible, client_disponible
    _TIENE_CLIENTE = True
except ImportError:
    _TIENE_CLIENTE = False
    def client_disponible(): return False
    class ClienteNoDisponible(Exception): pass


def _cliente():
    """Devuelve el cliente si la Pi está disponible, o None."""
    if not _TIENE_CLIENTE:
        return None
    try:
        return get_client(auto_descubrir=False)   # no bloquea si ya sabe que no hay
    except ClienteNoDisponible:
        return None


# ══════════════════════════════════════════════════════════════════════
#  INDICADOR DE FUENTE  (útil para mostrar en la UI)
# ══════════════════════════════════════════════════════════════════════

def fuente_activa() -> str:
    """
    Devuelve "api" si los datos vienen de la Raspberry,
    o "local" si vienen de la BD local.
    """
    c = _cliente()
    return "api" if (c and c.ping()) else "local"


# ══════════════════════════════════════════════════════════════════════
#  KPIs  (ResumenView)
# ══════════════════════════════════════════════════════════════════════

def kpi_dentro() -> int:
    """
    Personas actualmente dentro = entradas_hoy - salidas_hoy.
    No depende de usuario_id, funciona con salidas anónimas del botón.
    Nunca regresa negativo.
    """
    c = _cliente()
    if c:
        return c.resumen(n=0).get("kpi_dentro", 0)
    try:
        conn = _db()
        cur  = conn.cursor()
        cur.execute("""
            SELECT
                SUM(CASE WHEN tipo_evento = 'entrada' THEN 1 ELSE 0 END) -
                SUM(CASE WHEN tipo_evento = 'salida'  THEN 1 ELSE 0 END)
            FROM registro_acceso
            WHERE tipo_evento IN ('entrada', 'salida')
              AND DATE(fecha_acceso, 'localtime') = DATE('now', 'localtime')
        """)
        n = cur.fetchone()[0] or 0
        conn.close()
        return max(0, n)   # nunca negativo
    except Exception as e:
        print(f"[DS] kpi_dentro: {e}")
        return 0


def kpi_hoy() -> int:
    c = _cliente()
    if c:
        return c.resumen(n=0).get("kpi_hoy", 0)
    try:
        conn = _db(); cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM registro_acceso
            WHERE tipo_evento='entrada'
              AND DATE(fecha_acceso,'localtime')=DATE('now','localtime')
        """)
        n = cur.fetchone()[0]; conn.close(); return n
    except Exception as e:
        print(f"[DS] kpi_hoy: {e}"); return 0


def kpi_semana() -> int:
    c = _cliente()
    if c:
        return c.resumen(n=0).get("kpi_semana", 0)
    try:
        conn = _db(); cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM registro_acceso
            WHERE tipo_evento='entrada'
              AND DATE(fecha_acceso,'localtime') >= DATE('now','-6 days','localtime')
        """)
        n = cur.fetchone()[0]; conn.close(); return n
    except Exception as e:
        print(f"[DS] kpi_semana: {e}"); return 0


def ultimos_accesos(n: int = 4) -> list[dict]:
    c = _cliente()
    if c:
        data = c.resumen(n=n)
        return data.get("ultimos", [])
    # ── local ──
    try:
        conn = _db(); cur = conn.cursor()
        cur.execute("""
            SELECT nombre||' '||apellido_paterno,
                   numero_cuenta, rol, tipo_evento, fecha_acceso
            FROM registro_acceso ORDER BY id DESC LIMIT ?
        """, (n,))
        rows = cur.fetchall(); conn.close()
        return [{"nombre": (r[0] or "").strip(),
                 "cuenta": r[1] or "—",
                 "rol":    r[2],
                 "tipo":   r[3],
                 "hora":   (r[4] or "")[11:16]}
                for r in rows]
    except Exception as e:
        print(f"[DS] ultimos_accesos: {e}"); return []


# ══════════════════════════════════════════════════════════════════════
#  ACCESOS COMPLETOS  (AccesosView)
# ══════════════════════════════════════════════════════════════════════

def accesos_todos(rol_filtro: str = "") -> list[dict]:
    c = _cliente()
    if c:
        return c.accesos(rol=rol_filtro)
    # ── local ──
    try:
        conn = _db(); cur = conn.cursor()
        sql = """
            SELECT ra.nombre||' '||ra.apellido_paterno,
                   ra.numero_cuenta, ra.rol, ra.tipo_evento, ra.fecha_acceso,
                   ed.grado, ed.grupo
            FROM registro_acceso ra
            LEFT JOIN estudiantes_detalle ed ON ed.usuario_id = ra.usuario_id
        """
        params: tuple = ()
        if rol_filtro:
            sql    += " WHERE ra.rol=?"
            params  = (rol_filtro,)
        sql += " ORDER BY ra.id ASC"
        cur.execute(sql, params)
        rows = cur.fetchall(); conn.close()
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
        return result
    except Exception as e:
        print(f"[DS] accesos_todos: {e}"); return []


# ══════════════════════════════════════════════════════════════════════
#  ESTADÍSTICAS  (StatsView)
# ══════════════════════════════════════════════════════════════════════

def top7() -> list[dict]:
    c = _cliente()
    if c:
        return c.stats().get("top7", [])
    # ── local ──
    try:
        from dash_theme import iniciales as _ini
        conn = _db(); cur = conn.cursor()
        cur.execute("""
            SELECT ra.nombre||' '||ra.apellido_paterno,
                   ra.numero_cuenta, ra.rol, COUNT(*) AS vis,
                   ed.grado, ed.grupo
            FROM registro_acceso ra
            LEFT JOIN estudiantes_detalle ed ON ed.usuario_id = ra.usuario_id
            WHERE ra.tipo_evento='entrada'
            GROUP BY ra.usuario_id ORDER BY vis DESC LIMIT 7
        """)
        rows = cur.fetchall(); conn.close()
        result = []
        for nom, cuenta, rol, vis, grado, grupo in rows:
            nom   = (nom or "").strip()
            parts = nom.split(" ", 1)
            sub   = (rol.capitalize() if rol in ("maestro","admin") else
                     f"{grado}° {grupo}" if grado and grupo else
                     f"{grado}°"          if grado           else "—")
            result.append({
                "nombre":  nom,
                "cuenta":  cuenta or "—",
                "rol":     rol or "—",
                "visitas": vis,
                "sub":     sub,
                "ini":     _ini(parts[0], parts[1] if len(parts) > 1 else ""),
            })
        return result
    except Exception as e:
        print(f"[DS] top7: {e}"); return []


def stats_rol() -> list[tuple]:
    """Devuelve lista de (rol, cnt)."""
    c = _cliente()
    if c:
        raw = c.stats().get("por_rol", [])
        return [(r["rol"], r["cnt"]) for r in raw]
    try:
        conn = _db(); cur = conn.cursor()
        cur.execute("""
            SELECT rol, COUNT(*) FROM registro_acceso
            WHERE tipo_evento='entrada'
            GROUP BY rol ORDER BY COUNT(*) DESC
        """)
        rows = cur.fetchall(); conn.close(); return rows
    except Exception as e:
        print(f"[DS] stats_rol: {e}"); return []


def stats_hora() -> list[tuple]:
    """Devuelve lista de (hora_int, cnt)."""
    c = _cliente()
    if c:
        raw = c.stats().get("por_hora", [])
        return [(r["hora"], r["cnt"]) for r in raw]
    try:
        conn = _db(); cur = conn.cursor()
        cur.execute("""
            SELECT CAST(STRFTIME('%H',fecha_acceso,'localtime') AS INTEGER),
                   COUNT(*)
            FROM registro_acceso
            WHERE tipo_evento='entrada'
              AND DATE(fecha_acceso,'localtime') >= DATE('now','-6 days','localtime')
            GROUP BY 1 ORDER BY 1
        """)
        rows = cur.fetchall(); conn.close(); return rows
    except Exception as e:
        print(f"[DS] stats_hora: {e}"); return []


# ══════════════════════════════════════════════════════════════════════
#  PERFIL STATS  (PerfilView)
# ══════════════════════════════════════════════════════════════════════

def perfil_stats() -> tuple[int, int]:
    """Devuelve (accesos_mes, usuarios_mes)."""
    c = _cliente()
    if c:
        return c.perfil_stats()
    try:
        conn = _db(); cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM registro_acceso
            WHERE tipo_evento='entrada'
              AND DATE(fecha_acceso,'localtime') >=
                  DATE('now','start of month','localtime')
        """)
        acc = cur.fetchone()[0]
        cur.execute("""
            SELECT COUNT(*) FROM usuarios
            WHERE DATE(fecha_registro,'localtime') >=
                  DATE('now','start of month','localtime')
        """)
        reg = cur.fetchone()[0]
        conn.close(); return acc, reg
    except Exception as e:
        print(f"[DS] perfil_stats: {e}"); return 0, 0
    
def ultima_persona_acceso() -> dict | None:
    """
    Devuelve los datos de la última persona que registró una ENTRADA,
    junto con el total acumulado de sus entradas.
 
    Retorna un dict con claves:
        nombre, cuenta, rol, hora, fecha, visitas_total
    o None si no hay registros.
    """
    try:
        conn = _db()
        cur  = conn.cursor()
        cur.execute("""
            SELECT ra.usuario_id,
                   ra.nombre || ' ' || ra.apellido_paterno AS nombre,
                   ra.numero_cuenta,
                   ra.rol,
                   ra.fecha_acceso,
                   ra.detalle
            FROM registro_acceso ra
            WHERE ra.tipo_evento = 'entrada'
            ORDER BY ra.id DESC
            LIMIT 1
        """)
        row = conn.cursor().fetchone()
 
        # Re-ejecutar porque fetchone en cursor anterior ya avanzó
        cur.execute("""
            SELECT ra.usuario_id,
                   ra.nombre || ' ' || ra.apellido_paterno,
                   ra.numero_cuenta,
                   ra.rol,
                   ra.fecha_acceso,
                   ra.detalle
            FROM registro_acceso ra
            WHERE ra.tipo_evento = 'entrada'
            ORDER BY ra.id DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            conn.close()
            return None
 
        uid, nombre, cuenta, rol, fecha_acceso, detalle = row
 
        # Contar total de entradas de esa persona
        cur.execute("""
            SELECT COUNT(*) FROM registro_acceso
            WHERE usuario_id = ? AND tipo_evento = 'entrada'
        """, (uid,))
        total = cur.fetchone()[0]
        conn.close()
 
        return {
            "usuario_id":    uid,
            "nombre":        (nombre or "").strip(),
            "cuenta":        cuenta or "—",
            "rol":           rol or "—",
            "hora":          (fecha_acceso or "")[11:16] or "—",
            "fecha":         (fecha_acceso or "")[:10]   or "—",
            "visitas_total": total,
        }
    except Exception as e:
        print(f"[DS] ultima_persona_acceso: {e}")
        return None
 
 
def total_accesos_usuario(usuario_id: int) -> int:
    """
    Devuelve el número total de entradas registradas para un usuario_id.
    Útil para mostrar el contador en tiempo real.
    """
    try:
        conn = _db()
        cur  = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM registro_acceso
            WHERE usuario_id = ? AND tipo_evento = 'entrada'
        """, (usuario_id,))
        n = cur.fetchone()[0]
        conn.close()
        return n
    except Exception as e:
        print(f"[DS] total_accesos_usuario: {e}")
        return 0