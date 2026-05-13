"""
api_client.py  ─  Cliente HTTP  (corre en la laptop)
=====================================================
Sustituye las llamadas directas a database.py cuando el dashboard
está conectado a la Raspberry Pi via WiFi.

Expone las MISMAS funciones que se usan en dashboard_views.py
y dashboard_gestion.py, para que el resto del código no cambie.

Uso:
    from api_client import get_client, ClienteNoDisponible

    try:
        c = get_client()          # lanza excepción si no hay conexión
        datos = c.resumen()       # dict con kpi_dentro, kpi_hoy, etc.
    except ClienteNoDisponible:
        # fallback a BD local
        pass
"""

import json
import urllib.request
import urllib.error
import urllib.parse
import os
from typing import Optional

# ── Configuración ─────────────────────────────────────────────────────
_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             ".pi_config.json")
_DEFAULT_CFG = {
    "pi_ip":   "192.168.4.1",   # IP del hotspot de la Raspberry
    "pi_port": 5000,
    "token":   "labcontrol-secret-2025",
}
TIMEOUT = 6     # segundos por petición


class ClienteNoDisponible(Exception):
    """Se lanza cuando la Raspberry Pi no responde."""
    pass


def cargar_config() -> dict:
    try:
        with open(_CONFIG_FILE, "r") as f:
            return {**_DEFAULT_CFG, **json.load(f)}
    except:
        return _DEFAULT_CFG.copy()


def guardar_config(cfg: dict):
    try:
        with open(_CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[API_CLIENT] No se pudo guardar config: {e}")


# ══════════════════════════════════════════════════════════════════════
class LabControlClient:

    def __init__(self, ip: str, port: int, token: str):
        self.base  = f"http://{ip}:{port}"
        self.token = token

    # ── Bajo nivel ───────────────────────────────────────────────────

    def _headers(self):
        return {"Content-Type": "application/json",
                "X-LabControl-Token": self.token}

    def _get(self, path: str, params: dict = None) -> Optional[dict]:
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None})
        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"[API_CLIENT] GET {path} → {e}")
            return None

    def _post(self, path: str, body: dict) -> Optional[dict]:
        url  = self.base + path
        data = json.dumps(body, default=_json_safe).encode()
        try:
            req = urllib.request.Request(
                url, data=data, headers=self._headers(), method="POST")
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            try:
                err_body = json.loads(e.read().decode())
                print(f"[API_CLIENT] POST {path} HTTP {e.code}: "
                      f"{err_body.get('error','?')}")
            except:
                print(f"[API_CLIENT] POST {path} HTTP {e.code}")
            return None
        except Exception as e:
            print(f"[API_CLIENT] POST {path} → {e}")
            return None

    def _put(self, path: str, body: dict) -> Optional[dict]:
        url  = self.base + path
        data = json.dumps(body, default=_json_safe).encode()
        try:
            req = urllib.request.Request(
                url, data=data, headers=self._headers(), method="PUT")
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"[API_CLIENT] PUT {path} → {e}")
            return None

    def _delete(self, path: str) -> Optional[dict]:
        url = self.base + path
        try:
            req = urllib.request.Request(
                url, headers=self._headers(), method="DELETE")
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"[API_CLIENT] DELETE {path} → {e}")
            return None

    # ── Health ───────────────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base}/ping")
            with urllib.request.urlopen(req, timeout=3) as r:
                return json.loads(r.read().decode()).get("ok") is True
        except:
            return False

    # ── Datos para ResumenView ────────────────────────────────────────

    def resumen(self, n: int = 4) -> dict:
        """
        Devuelve:
            kpi_dentro, kpi_hoy, kpi_semana, ultimos (lista)
        """
        r = self._get("/resumen", {"n": n})
        if r and r.get("ok"):
            return r["data"]
        return {"kpi_dentro": 0, "kpi_hoy": 0,
                "kpi_semana": 0, "ultimos": []}

    # ── Datos para AccesosView ────────────────────────────────────────

    def accesos(self, rol: str = "", limit: int = 500) -> list:
        """Lista de accesos ordenada ASC. Igual a _accesos_todos()."""
        r = self._get("/accesos", {"rol": rol, "limit": limit})
        if r and r.get("ok"):
            return r["data"]
        return []

    # ── Datos para StatsView ─────────────────────────────────────────

    def stats(self) -> dict:
        """
        Devuelve:
            top7     → lista de dicts (nombre, cuenta, rol, visitas, sub, ini)
            por_rol  → lista de dicts (rol, cnt)
            por_hora → lista de dicts (hora, cnt)
        """
        r = self._get("/stats")
        if r and r.get("ok"):
            return r["data"]
        return {"top7": [], "por_rol": [], "por_hora": []}

    # ── Datos para PerfilView ─────────────────────────────────────────

    def perfil_stats(self) -> tuple[int, int]:
        """Devuelve (accesos_mes, usuarios_mes)."""
        r = self._get("/perfil/stats")
        if r and r.get("ok"):
            return r["data"]["accesos_mes"], r["data"]["usuarios_mes"]
        return 0, 0

    # ── Usuarios CRUD (GestionView) ───────────────────────────────────

    def listar_usuarios(self, rol: str = None) -> list:
        r = self._get("/usuarios", {"rol": rol})
        if r and r.get("ok"):
            return r["data"]
        return []

    def obtener_usuario(self, uid: int) -> Optional[dict]:
        r = self._get(f"/usuarios/{uid}")
        if r and r.get("ok"):
            return r["data"]
        return None

    def registrar_usuario(self,
                           nombre: str,
                           apellido_paterno: str,
                           apellido_materno: str = "",
                           rol:  str = "estudiante",
                           numero_cuenta: str = None,
                           correo: str = None,
                           contrasena: str = None,
                           grado: int  = None,
                           grupo: str  = None,
                           registrado_por: int = None,
                           **_) -> int:
        body = {
            "nombre":           nombre,
            "apellido_paterno": apellido_paterno,
            "apellido_materno": apellido_materno,
            "rol":              rol,
            "numero_cuenta":    numero_cuenta,
            "correo":           correo,
            "contrasena":       contrasena,
            "grado":            grado,
            "grupo":            grupo,
            "registrado_por":   registrado_por,
        }
        r = self._post("/usuarios", body)
        if r and r.get("ok"):
            return r["data"]["id"]
        return -1

    def editar_usuario(self, uid: int, datos: dict,
                        editor_id: int = None) -> tuple[bool, str]:
        payload = {**datos}
        if editor_id:
            payload["actualizado_por"] = editor_id
        r = self._put(f"/usuarios/{uid}", payload)
        if r and r.get("ok"):
            return True, "Éxito"
        return False, "No se pudo actualizar."

    def eliminar_usuario(self, uid: int) -> bool:
        r = self._delete(f"/usuarios/{uid}")
        return bool(r and r.get("ok"))

    # Alias
    eliminar_persona = eliminar_usuario

    # ── Biometría ─────────────────────────────────────────────────────

    def guardar_vectores_por_angulo(self,
                                     uid: int,
                                     vectores_por_paso: dict) -> int:
        import numpy as np
        vec_json = {}
        for ang, datos in vectores_por_paso.items():
            vecs = datos.get("vectores", [])
            if not vecs:
                continue
            vec_json[ang] = {
                "vectores": [
                    v.tolist() if isinstance(v, np.ndarray) else list(v)
                    for v in vecs
                ]
            }
        r = self._post(f"/usuarios/{uid}/vectores",
                        {"vectores_angulo": vec_json})
        if r and r.get("ok"):
            return r["data"].get("angulos_guardados", 0)
        return 0

    def tiene_vector(self, uid: int) -> bool:
        r = self._get(f"/usuarios/{uid}/vectores")
        return bool(r and r.get("ok") and r["data"].get("tiene_vector"))

    def angulos_registrados(self, uid: int) -> list:
        r = self._get(f"/usuarios/{uid}/vectores")
        if r and r.get("ok"):
            return r["data"].get("angulos", [])
        return []

    # ── Login (para cuando la laptop quiera autenticar contra la Pi) ──

    def login(self, usuario: str, contrasena: str) -> Optional[dict]:
        """
        Devuelve el dict del usuario si las credenciales son válidas,
        o None si son incorrectas.
        """
        r = self._post("/login",
                        {"usuario": usuario, "contrasena": contrasena})
        if r and r.get("ok"):
            return r["data"]
        return None

    # ── Sync ──────────────────────────────────────────────────────────

    def sync_push(self, usuario: dict, vectores_angulo: dict = None) -> dict:
        import numpy as np
        vec_json = {}
        if vectores_angulo:
            for ang, datos in vectores_angulo.items():
                vecs = datos.get("vectores", [])
                if not vecs:
                    continue
                vec_json[ang] = {
                    "vectores": [
                        v.tolist() if isinstance(v, np.ndarray) else list(v)
                        for v in vecs
                    ]
                }
        r = self._post("/sync/push",
                        {"usuario": usuario, "vectores_angulo": vec_json})
        return r.get("data", {}) if r and r.get("ok") else {}

    def sync_export(self) -> Optional[dict]:
        r = self._get("/sync/export")
        if r and r.get("ok"):
            return r["data"]
        return None


# ════════════════════════════════════════════════════════════════════
#  DESCUBRIMIENTO AUTOMÁTICO
# ════════════════════════════════════════════════════════════════════

def _json_safe(obj):
    """Serializa numpy arrays a lista."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass
    raise TypeError(f"No serializable: {type(obj)}")


def descubrir_raspberry(
        ip_candidata: str = None,
        puerto:  int  = None,
        token:   str  = None,
) -> Optional[LabControlClient]:
    """
    Intenta conectar con la Raspberry Pi.
    Primero prueba la IP guardada en .pi_config.json; si falla,
    escanea las IPs .1–.10 de la subred del hotspot (192.168.4.x).
    """
    cfg = cargar_config()
    ip  = ip_candidata or cfg["pi_ip"]
    prt = puerto       or cfg["pi_port"]
    tok = token        or cfg["token"]

    c = LabControlClient(ip, prt, tok)
    if c.ping():
        print(f"[API_CLIENT] Pi encontrada en {ip}:{prt}")
        cfg["pi_ip"] = ip
        guardar_config(cfg)
        return c

    # Escaneo de subred
    subred = ".".join(ip.split(".")[:3])
    for i in range(1, 11):
        ip_scan = f"{subred}.{i}"
        if ip_scan == ip:
            continue
        c2 = LabControlClient(ip_scan, prt, tok)
        if c2.ping():
            print(f"[API_CLIENT] Pi encontrada en {ip_scan}:{prt}")
            cfg["pi_ip"] = ip_scan
            guardar_config(cfg)
            return c2

    print("[API_CLIENT] Raspberry Pi no encontrada en la red.")
    return None


# ════════════════════════════════════════════════════════════════════
#  INSTANCIA GLOBAL  (singleton por sesión)
# ════════════════════════════════════════════════════════════════════

_client_instance: Optional[LabControlClient] = None


def get_client(auto_descubrir: bool = True) -> LabControlClient:
    """
    Devuelve el cliente activo o intenta descubrir la Pi.
    Lanza ClienteNoDisponible si no hay conexión.
    """
    global _client_instance
    if _client_instance and _client_instance.ping():
        return _client_instance
    if auto_descubrir:
        _client_instance = descubrir_raspberry()
    if not _client_instance:
        raise ClienteNoDisponible(
            "No se pudo conectar con la Raspberry Pi.")
    return _client_instance


def set_client(c: LabControlClient):
    """Inyectar un cliente externo (útil para tests)."""
    global _client_instance
    _client_instance = c


def client_disponible() -> bool:
    """True si la Pi está accesible en este momento."""
    try:
        get_client()
        return True
    except ClienteNoDisponible:
        return False