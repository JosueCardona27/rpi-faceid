"""
sensores_ir.py  —  Raspberry Pi 5  (FC-51 x2, entrada y salida)
================================================================================
Detecta entrada y salida de personas con dos sensores FC-51 y registra
el evento directamente en la base de datos (tabla registro_acceso).

  • Sensor ENTRADA → GPIO 5  (pin físico 29)   [lado EXTERIOR de la puerta]
  • Sensor SALIDA  → GPIO 6  (pin físico 31)   [lado INTERIOR de la puerta]

LÓGICA DE SECUENCIA:
  - Entrada: S1 (exterior) dispara PRIMERO, luego S2 (interior) → entrada
  - Salida : S2 (interior) dispara PRIMERO, luego S1 (exterior) → salida
  - Un solo sensor dispara sin que el otro responda → ignorado (falso positivo)

  - FC-51 en reposo    → OUT = HIGH  (sin obstáculo)
  - FC-51 con persona  → OUT = LOW   (obstáculo detectado)

CONEXIÓN FC-51:
  VCC  → Pin 1  (3.3V)
  GND  → Pin 39 (GND)
  OUT  → GPIO correspondiente

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
PIN_ENTRADA = 5    # pin físico 29  — sensor EXTERIOR
PIN_SALIDA  = 6    # pin físico 31  — sensor INTERIOR

# ── Configuración ─────────────────────────────────────────────────────────────
DEBOUNCE_S        = 0.3    # segundos mínimos entre detecciones del mismo sensor
TIMEOUT_SEQ_S     = 3.0    # segundos máximos entre S1 y S2 para contar el par
USUARIO_IR_NOMBRE = "Sensor"
USUARIO_IR_CUENTA = "00000000"


# ══════════════════════════════════════════════════════════════════════════════
def _obtener_o_crear_usuario_ir() -> int | None:
    if not _DB_OK:
        return None
    try:
        conn   = conectar()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM usuarios WHERE numero_cuenta = ?",
            (USUARIO_IR_CUENTA,))
        row = cursor.fetchone()
        if row:
            conn.close()
            return row[0]
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
    Controla dos sensores FC-51 con detección de secuencia.
    S1→S2 = entrada  |  S2→S1 = salida
    """

    def __init__(self, on_entrada=None, on_salida=None):
        self._on_entrada   = on_entrada or (lambda uid: None)
        self._on_salida    = on_salida  or (lambda uid: None)
        self._handle       = None
        self._running      = False
        self._hilo_entrada = None
        self._hilo_salida  = None
        self._usuario_id   = None

        # Debounce individual por sensor
        self._t_ultimo_entrada = 0.0
        self._t_ultimo_salida  = 0.0

        # ── Detección de secuencia ────────────────────────────────────────
        # Guardamos cuándo disparó cada sensor por última vez.
        # El segundo sensor en disparar es quien decide si fue entrada o salida.
        self._t_disparo_s1 = 0.0   # cuándo disparó S1 (exterior)
        self._t_disparo_s2 = 0.0   # cuándo disparó S2 (interior)
        self._seq_lock     = threading.Lock()   # protege ambos timestamps
        # ─────────────────────────────────────────────────────────────────

        # Contador en memoria
        self._personas_dentro = 0
        self._lock_contador   = threading.Lock()

        if _GPIO_OK:
            self._setup_gpio()
        if _DB_OK:
            self._usuario_id = _obtener_o_crear_usuario_ir()

    # ── GPIO ──────────────────────────────────────────────────────────────────
    def _setup_gpio(self):
        try:
            self._handle = lgpio.gpiochip_open(4)
        except Exception:
            self._handle = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_input(self._handle, PIN_ENTRADA)
        lgpio.gpio_claim_input(self._handle, PIN_SALIDA)
        print(f"[IR] Sensores listos — S1(exterior) GPIO{PIN_ENTRADA} | S2(interior) GPIO{PIN_SALIDA}")

    # ── Registro en BD ────────────────────────────────────────────────────────
    def _registrar_en_bd(self, tipo: str):
        if not _DB_OK or self._usuario_id is None:
            print(f"[IR] BD no disponible — evento {tipo} no registrado")
            return
        ok = registrar_acceso(self._usuario_id, tipo)
        if ok:
            print(f"[IR] {tipo.upper()} registrada en BD (usuario_id={self._usuario_id})")
        else:
            print(f"[IR] Error al registrar {tipo} en BD")

    # ── Monitor S1 — sensor EXTERIOR ─────────────────────────────────────────
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
                                tiempo_desde_s2 = ahora - self._t_disparo_s2

                                # S2 disparó ANTES que S1 dentro del timeout → SALIDA
                                if 0 < tiempo_desde_s2 < TIMEOUT_SEQ_S:
                                    self._t_disparo_s1 = 0.0
                                    self._t_disparo_s2 = 0.0
                                    procesar = "salida"
                                else:
                                    # S1 disparó solo: guardamos timestamp y esperamos a S2
                                    procesar = None

                            if procesar == "salida":
                                with self._lock_contador:
                                    self._personas_dentro = max(0, self._personas_dentro - 1)
                                    dentro = self._personas_dentro
                                print(f"[IR] SALIDA (S2→S1) — personas dentro: {dentro}")
                                self._registrar_en_bd("salida")
                                try:
                                    self._on_salida(self._usuario_id)
                                except Exception as e:
                                    print(f"[IR] Error callback salida: {e}")

                estado_anterior = lectura
            except Exception as e:
                print(f"[IR] Error sensor S1: {e}")
            time.sleep(0.02)

    # ── Monitor S2 — sensor INTERIOR ─────────────────────────────────────────
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
                                tiempo_desde_s1 = ahora - self._t_disparo_s1

                                # S1 disparó ANTES que S2 dentro del timeout → ENTRADA
                                if 0 < tiempo_desde_s1 < TIMEOUT_SEQ_S:
                                    self._t_disparo_s1 = 0.0
                                    self._t_disparo_s2 = 0.0
                                    procesar = "entrada"
                                else:
                                    # S2 disparó solo: guardamos timestamp y esperamos a S1
                                    procesar = None

                            if procesar == "entrada":
                                with self._lock_contador:
                                    self._personas_dentro += 1
                                    dentro = self._personas_dentro
                                print(f"[IR] ENTRADA (S1→S2) — personas dentro: {dentro}")
                                self._registrar_en_bd("entrada")
                                try:
                                    self._on_entrada(self._usuario_id)
                                except Exception as e:
                                    print(f"[IR] Error callback entrada: {e}")

                estado_anterior = lectura
            except Exception as e:
                print(f"[IR] Error sensor S2: {e}")
            time.sleep(0.02)

    # ── API pública ───────────────────────────────────────────────────────────
    def iniciar(self):
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
        print("[IR] Monitoreo iniciado — secuencia S1→S2=entrada / S2→S1=salida")

    def detener(self):
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
        with self._lock_contador:
            return self._personas_dentro

    def resetear_contador(self):
        with self._lock_contador:
            self._personas_dentro = 0
        print("[IR] Contador reseteado a 0")


# ══════════════════════════════════════════════════════════════════════════════
#  PRUEBA INDEPENDIENTE — python3 sensores_ir.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Prueba Sensores FC-51 (secuencia) ===")
    print(f"Cruza frente a S1(GPIO{PIN_ENTRADA}) luego S2(GPIO{PIN_SALIDA}) para ENTRADA")
    print(f"Cruza frente a S2(GPIO{PIN_SALIDA}) luego S1(GPIO{PIN_ENTRADA}) para SALIDA")
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