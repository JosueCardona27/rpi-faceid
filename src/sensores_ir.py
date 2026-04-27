"""
sensores_ir.py  —  Raspberry Pi 5  (FC-51 x2, entrada y salida)
================================================================================
Detecta entrada y salida de personas con dos sensores FC-51 y registra
el evento directamente en la base de datos (tabla registro_acceso).

  • Sensor ENTRADA → GPIO 5  (pin físico 29)
  • Sensor SALIDA  → GPIO 6  (pin físico 31)

LÓGICA:
  - FC-51 en reposo    → OUT = HIGH  (sin obstáculo)
  - FC-51 con persona  → OUT = LOW   (obstáculo detectado)

  - Sensor ENTRADA detecta → registra tipo_evento='entrada' en BD
  - Sensor SALIDA  detecta → registra tipo_evento='salida'  en BD

CONEXIÓN FC-51:
  VCC  → Pin 1  (3.3V)   ← FC-51 funciona con 3.3V, sin divisor necesario
  GND  → Pin 39 (GND)
  OUT  → GPIO correspondiente

NOTA: El FC-51 necesita que el registro_acceso tenga un usuario_id válido.
      Este archivo usa un usuario_id especial llamado "Conteo General" que
      se crea automáticamente si no existe en la BD.

Dependencias:
  pip install lgpio --break-system-packages
"""

import time
import threading
import os
import sys

# ── Agregar src/ al path para importar database.py ───────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# ── Intentar importar lgpio ───────────────────────────────────────────────────
try:
    import lgpio
    _GPIO_OK = True
except ImportError:
    _GPIO_OK = False
    print("[IR] lgpio no disponible — modo simulado activado")

# ── Intentar importar database ────────────────────────────────────────────────
try:
    from database import registrar_acceso, conectar
    _DB_OK = True
except ImportError as e:
    print(f"[IR] database no disponible: {e}")
    _DB_OK = False

# ── Pines BCM ─────────────────────────────────────────────────────────────────
PIN_ENTRADA = 5    # pin físico 29
PIN_SALIDA  = 6    # pin físico 31

# ── Configuración ─────────────────────────────────────────────────────────────
DEBOUNCE_S         = 0.3    # segundos de espera entre detecciones del mismo sensor
USUARIO_IR_NOMBRE  = "Sensor"
USUARIO_IR_CUENTA  = "00000000"   # número de cuenta especial para el sensor


# ══════════════════════════════════════════════════════════════════════════════
def _obtener_o_crear_usuario_ir() -> int | None:
    """
    Busca o crea el usuario especial del sensor IR en la BD.
    Devuelve su usuario_id o None si falla.
    """
    if not _DB_OK:
        return None
    try:
        conn   = conectar()
        cursor = conn.cursor()

        # Buscar si ya existe
        cursor.execute(
            "SELECT id FROM usuarios WHERE numero_cuenta = ?",
            (USUARIO_IR_CUENTA,))
        row = cursor.fetchone()
        if row:
            conn.close()
            return row[0]

        # Crear usuario especial para el sensor
        cursor.execute("""
            INSERT INTO usuarios
                (nombre, apellido_paterno, apellido_materno,
                 numero_cuenta, rol)
            VALUES (?, ?, ?, ?, ?)
        """, (USUARIO_IR_NOMBRE, "IR", "Conteo",
              USUARIO_IR_CUENTA, "estudiante"))
        uid = cursor.lastrowid
        conn.commit()
        conn.close()
        print(f"[IR] Usuario sensor creado ID={uid}")
        return uid

    except Exception as e:
        print(f"[IR] Error creando usuario sensor: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
class SensoresIR:
    """
    Controla dos sensores FC-51 para registrar entradas y salidas.

    Uso desde interfaz.py o independiente:
        from sensores_ir import SensoresIR
        sensores = SensoresIR()
        sensores.iniciar()

        # Al cerrar:
        sensores.detener()
    """

    def __init__(self, on_entrada=None, on_salida=None):
        """
        on_entrada : callback opcional cuando se detecta una entrada
        on_salida  : callback opcional cuando se detecta una salida
        """
        self._on_entrada   = on_entrada or (lambda uid: None)
        self._on_salida    = on_salida  or (lambda uid: None)
        self._handle       = None
        self._running      = False
        self._hilo_entrada = None
        self._hilo_salida  = None
        self._usuario_id   = None

        # Timestamps para debounce independiente por sensor
        self._t_ultimo_entrada = 0.0
        self._t_ultimo_salida  = 0.0

        self._t_disparo_s1 = 0.0   # cuándo disparó el sensor exterior
        self._t_disparo_s2 = 0.0   # cuándo disparó el sensor interior
        self._seq_lock     = threading.Lock()
        self._TIMEOUT_SEQ  = 2.0

        # Contador en memoria (útil para mostrar en pantalla)
        self._personas_dentro = 0
        self._lock_contador   = threading.Lock()

        if _GPIO_OK:
            self._setup_gpio()

        if _DB_OK:
            self._usuario_id = _obtener_o_crear_usuario_ir()

    # ── GPIO ──────────────────────────────────────────────────────────────────
    def _setup_gpio(self):
        try:
            self._handle = lgpio.gpiochip_open(4)   # gpiochip4 en Pi 5
        except Exception:
            self._handle = lgpio.gpiochip_open(0)

        # FC-51 OUT como entrada, sin pull (el módulo ya tiene resistencia)
        lgpio.gpio_claim_input(self._handle, PIN_ENTRADA)
        lgpio.gpio_claim_input(self._handle, PIN_SALIDA)

        print(f"[IR] Sensores listos — ENTRADA GPIO{PIN_ENTRADA} | SALIDA GPIO{PIN_SALIDA}")

    # ── Registro en BD ────────────────────────────────────────────────────────
    def _registrar_en_bd(self, tipo: str):
        """Registra entrada o salida en la base de datos."""
        if not _DB_OK or self._usuario_id is None:
            print(f"[IR] BD no disponible — evento {tipo} no registrado")
            return
        ok = registrar_acceso(self._usuario_id, tipo)
        if ok:
            print(f"[IR] {tipo.upper()} registrada en BD (usuario_id={self._usuario_id})")
        else:
            print(f"[IR] Error al registrar {tipo} en BD")

    # ── Monitor entrada ───────────────────────────────────────────────────────
    def _monitor_entrada(self):
        estado_anterior = 1
        while self._running:
            try:
                lectura = lgpio.gpio_read(self._handle, PIN_ENTRADA)

                if lectura == 0 and estado_anterior == 1:
                    ahora = time.time()
                    if ahora - self._t_ultimo_entrada >= DEBOUNCE_S:
                        self._t_ultimo_entrada = ahora
                        time.sleep(0.05)

                        if lgpio.gpio_read(self._handle, PIN_ENTRADA) == 0:
                            with self._seq_lock:
                                self._t_disparo_s1 = ahora
                                # ¿S2 (interior) disparó ANTES que S1? → SALIDA
                                if 0 < (ahora - self._t_disparo_s2) < self._TIMEOUT_SEQ:
                                    self._t_disparo_s1 = self._t_disparo_s2 = 0.0
                                    tipo = "salida"
                                    with self._lock_contador:
                                        self._personas_dentro = max(0, self._personas_dentro - 1)
                                        dentro = self._personas_dentro
                                    print(f"[IR] SALIDA (S2→S1) — dentro: {dentro}")
                                    self._registrar_en_bd(tipo)
                                    try:
                                        self._on_salida(self._usuario_id)
                                    except Exception as e:
                                        print(f"[IR] Error callback salida: {e}")
                                # Si S2 no disparó antes, solo guardamos timestamp
                                # y esperamos a que S2 dispare para confirmar ENTRADA

                estado_anterior = lectura
            except Exception as e:
                print(f"[IR] Error sensor entrada: {e}")
            time.sleep(0.02)

    # ── Monitor salida ────────────────────────────────────────────────────────
    def _monitor_salida(self):
        estado_anterior = 1
        while self._running:
            try:
                lectura = lgpio.gpio_read(self._handle, PIN_SALIDA)

                if lectura == 0 and estado_anterior == 1:
                    ahora = time.time()
                    if ahora - self._t_ultimo_salida >= DEBOUNCE_S:
                        self._t_ultimo_salida = ahora
                        time.sleep(0.05)

                        if lgpio.gpio_read(self._handle, PIN_SALIDA) == 0:
                            with self._seq_lock:
                                self._t_disparo_s2 = ahora
                                # ¿S1 (exterior) disparó ANTES que S2? → ENTRADA
                                if 0 < (ahora - self._t_disparo_s1) < self._TIMEOUT_SEQ:
                                    self._t_disparo_s1 = self._t_disparo_s2 = 0.0
                                    tipo = "entrada"
                                    with self._lock_contador:
                                        self._personas_dentro += 1
                                        dentro = self._personas_dentro
                                    print(f"[IR] ENTRADA (S1→S2) — dentro: {dentro}")
                                    self._registrar_en_bd(tipo)
                                    try:
                                        self._on_entrada(self._usuario_id)
                                    except Exception as e:
                                        print(f"[IR] Error callback entrada: {e}")
                                # Si S1 no disparó antes, solo guardamos timestamp
                                # y esperamos a que S1 dispare para confirmar SALIDA

                estado_anterior = lectura
            except Exception as e:
                print(f"[IR] Error sensor salida: {e}")
            time.sleep(0.02)

    # ── API pública ───────────────────────────────────────────────────────────
    def iniciar(self):
        """Inicia el monitoreo de ambos sensores en hilos separados."""
        if not _GPIO_OK:
            print("[IR] GPIO no disponible — sensores no iniciados")
            return
        self._running = True
        self._hilo_entrada = threading.Thread(
            target=self._monitor_entrada, daemon=True)
        self._hilo_salida  = threading.Thread(
            target=self._monitor_salida, daemon=True)
        self._hilo_entrada.start()
        self._hilo_salida.start()
        print("[IR] Monitoreo de sensores iniciado")

    def detener(self):
        """Detiene el monitoreo y libera GPIO."""
        print("[IR] Deteniendo sensores...")
        self._running = False
        if self._hilo_entrada:
            self._hilo_entrada.join(timeout=2)
        if self._hilo_salida:
            self._hilo_salida.join(timeout=2)
        if self._handle is not None:
            try:
                lgpio.gpiochip_close(self._handle)
            except Exception:
                pass
        print("[IR] Sensores detenidos.")

    @property
    def personas_dentro(self) -> int:
        """Número actual de personas dentro según el conteo en memoria."""
        with self._lock_contador:
            return self._personas_dentro

    def resetear_contador(self):
        """Resetea el contador a 0 (útil al inicio del día)."""
        with self._lock_contador:
            self._personas_dentro = 0
        print("[IR] Contador reseteado a 0")


# ══════════════════════════════════════════════════════════════════════════════
#  PRUEBA INDEPENDIENTE
#  python3 sensores_ir.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Prueba Sensores FC-51 ===")
    print("Pasa la mano frente a cada sensor")
    print("Presiona Ctrl+C para salir\n")

    def entrada_detectada(uid):
        print(f">>> ENTRADA registrada (uid={uid})")

    def salida_detectada(uid):
        print(f">>> SALIDA registrada (uid={uid})")

    sensores = SensoresIR(
        on_entrada=entrada_detectada,
        on_salida=salida_detectada
    )
    sensores.iniciar()

    try:
        while True:
            time.sleep(1)
            print(f"Personas dentro: {sensores.personas_dentro}", end="\r")
    except KeyboardInterrupt:
        sensores.detener()
        print("\nPrueba finalizada.")
