"""
servo_control.py  —  Raspberry Pi 5  (usa lgpio, compatible con Pi 5)
================================================================================
Controla directamente desde los GPIO de la Raspberry Pi:
  • Servomotor  → GPIO 18  (PWM hardware, pin físico 12)
  • LED VERDE   → GPIO 23  (pin físico 16)
  • LED ROJO    → GPIO 24  (pin físico 18)
  • Buzzer      → GPIO 25  (pin físico 22)
  • Botón man.  → GPIO 17  (pin físico 11)  [INPUT con pull-up interno]

Dependencias:
  pip install lgpio --break-system-packages
  pip install gpiozero --break-system-packages
"""

import time
import threading

# ── Intentar importar lgpio ───────────────────────────────────────────────────
try:
    import lgpio
    _GPIO_OK = True
except ImportError:
    _GPIO_OK = False
    print("[HW] lgpio no disponible — modo simulado activado")


# ── Configuración de pines (BCM) ──────────────────────────────────────────────
PIN_SERVO     = 18
PIN_LED_VERDE = 23
PIN_LED_ROJO  = 24
PIN_BUZZER    = 25
PIN_BOTON     = 17

# ── Ángulos del servo ─────────────────────────────────────────────────────────
SERVO_ABIERTO  = 90
SERVO_CERRADO  =  0

# Conversión de ángulo → ancho de pulso en microsegundos
# 0°   → 500 µs
# 90°  → 1500 µs
# 180° → 2500 µs
def _angulo_a_pulso(grados: float) -> int:
    return int(500 + (grados / 180.0) * 2000)


# ── Tiempos (segundos) ────────────────────────────────────────────────────────
T_PUERTA_ABIERTA = 4.0
T_LED_DENEGADO   = 3.0
T_PARPADEO       = 0.3
T_BUZZER_ACCESO  = 0.2
T_BEEP           = 0.3
DEBOUNCE_S       = 0.05


# ══════════════════════════════════════════════════════════════════════════════
class _ControladorHW:

    def __init__(self, on_salida_manual=None):
        self._lock           = threading.Lock()
        self._handle         = None   # handle lgpio
        self._puerta_timer   = None
        self._running        = True

        self._setup_gpio()
        self._iniciar_monitor_boton()

    # ── Inicialización GPIO ───────────────────────────────────────────────────
    def _setup_gpio(self):
        # Abrir chip GPIO (gpiochip4 en Raspberry Pi 5)
        try:
            self._handle = lgpio.gpiochip_open(4)
        except Exception:
            self._handle = lgpio.gpiochip_open(0)

        # Salidas
        lgpio.gpio_claim_output(self._handle, PIN_LED_VERDE, 0)
        lgpio.gpio_claim_output(self._handle, PIN_LED_ROJO,  0)
        lgpio.gpio_claim_output(self._handle, PIN_BUZZER,    0)
        lgpio.gpio_claim_output(self._handle, PIN_SERVO,     0)

        # Entrada con pull-up para el botón
        lgpio.gpio_claim_input(self._handle, PIN_BOTON, lgpio.SET_PULL_UP)

        # Servo en posición cerrada al inicio
        self._mover_servo(SERVO_CERRADO)

        print("[HW] GPIO inicializado correctamente (lgpio)")
        print(f"[HW] Servo GPIO{PIN_SERVO} | LED Verde GPIO{PIN_LED_VERDE} "
              f"| LED Rojo GPIO{PIN_LED_ROJO} | Buzzer GPIO{PIN_BUZZER} "
              f"| Botón GPIO{PIN_BOTON}")

    # ── Control del servo ─────────────────────────────────────────────────────
    def _mover_servo(self, grados: float):
        pulso = _angulo_a_pulso(grados)
        lgpio.tx_servo(self._handle, PIN_SERVO, pulso)
        time.sleep(0.5)   # tiempo para que el servo llegue a la posición
        lgpio.tx_servo(self._handle, PIN_SERVO, 0)   # detener señal (evita vibración)

    # ── Buzzer ────────────────────────────────────────────────────────────────
    def _beep_acceso(self):
        lgpio.gpio_write(self._handle, PIN_BUZZER, 1)
        time.sleep(T_BUZZER_ACCESO)
        lgpio.gpio_write(self._handle, PIN_BUZZER, 0)

    def _triple_beep(self):
        def _run():
            for _ in range(3):
                if not self._running:
                    break
                lgpio.gpio_write(self._handle, PIN_BUZZER, 1)
                time.sleep(T_BEEP)
                lgpio.gpio_write(self._handle, PIN_BUZZER, 0)
                time.sleep(T_BEEP)
        threading.Thread(target=_run, daemon=True).start()

    # ── Parpadeo LED rojo ─────────────────────────────────────────────────────
    def _parpadeo_rojo(self):
        def _run():
            t_fin = time.time() + T_LED_DENEGADO
            while time.time() < t_fin and self._running:
                lgpio.gpio_write(self._handle, PIN_LED_ROJO, 1)
                time.sleep(T_PARPADEO)
                lgpio.gpio_write(self._handle, PIN_LED_ROJO, 0)
                time.sleep(T_PARPADEO)
            lgpio.gpio_write(self._handle, PIN_LED_ROJO, 0)
        threading.Thread(target=_run, daemon=True).start()

    # ── Cierre automático ─────────────────────────────────────────────────────
    def _programar_cierre(self):
        if self._puerta_timer:
            self._puerta_timer.cancel()
        self._puerta_timer = threading.Timer(T_PUERTA_ABIERTA, self._cerrar_puerta)
        self._puerta_timer.daemon = True
        self._puerta_timer.start()

    def _cerrar_puerta(self):
        with self._lock:
            self._mover_servo(SERVO_CERRADO)
            lgpio.gpio_write(self._handle, PIN_LED_VERDE, 0)
        print("[HW] Puerta cerrada (auto)")

    # ── Botón manual ──────────────────────────────────────────────────────────
    def _iniciar_monitor_boton(self):
        def _monitor():
            estado_anterior = 1
            while self._running:
                try:
                    lectura = lgpio.gpio_read(self._handle, PIN_BOTON)
                    if lectura == 0 and estado_anterior == 1:
                        time.sleep(DEBOUNCE_S)
                        if lgpio.gpio_read(self._handle, PIN_BOTON) == 0:
                            print("[HW] Botón presionado — apertura manual / salida")
                            self.abrir("Manual")
                            try:
                                self._on_salida_manual()   # ← registra la salida en BD
                            except Exception as e:
                                print(f"[HW] Error en callback salida: {e}")
                    estado_anterior = lectura
                except Exception:
                    pass
                time.sleep(0.02)
        threading.Thread(target=_monitor, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  API PÚBLICA
    # ══════════════════════════════════════════════════════════════════════════

    def abrir(self, nombre: str = ""):
        print(f"[HW] ACCESO PERMITIDO — {nombre}")
        with self._lock:
            lgpio.gpio_write(self._handle, PIN_LED_ROJO,  0)
            lgpio.gpio_write(self._handle, PIN_LED_VERDE, 1)
            self._mover_servo(SERVO_ABIERTO)
        threading.Thread(target=self._beep_acceso, daemon=True).start()
        self._programar_cierre()

    def denegar(self):
        print("[HW] ACCESO DENEGADO")
        with self._lock:
            lgpio.gpio_write(self._handle, PIN_LED_VERDE, 0)
            self._mover_servo(SERVO_CERRADO)
        self._parpadeo_rojo()
        self._triple_beep()

    def espera(self):
        print("[HW] MODO ESPERA")
        if self._puerta_timer:
            self._puerta_timer.cancel()
        with self._lock:
            lgpio.gpio_write(self._handle, PIN_LED_VERDE, 0)
            lgpio.gpio_write(self._handle, PIN_LED_ROJO,  0)
            lgpio.gpio_write(self._handle, PIN_BUZZER,    0)
            self._mover_servo(SERVO_CERRADO)

    def desconectar(self):
        print("[HW] Desconectando GPIO...")
        self._running = False
        if self._puerta_timer:
            self._puerta_timer.cancel()
        try:
            lgpio.gpio_write(self._handle, PIN_LED_VERDE, 0)
            lgpio.gpio_write(self._handle, PIN_LED_ROJO,  0)
            lgpio.gpio_write(self._handle, PIN_BUZZER,    0)
            lgpio.tx_servo(self._handle, PIN_SERVO, 0)
            lgpio.gpiochip_close(self._handle)
        except Exception:
            pass
        print("[HW] GPIO liberado.")


# ══════════════════════════════════════════════════════════════════════════════
class _ServoStub:
    def abrir(self, nombre: str = ""):
        print(f"[STUB] abrir() — {nombre}")
    def denegar(self):
        print("[STUB] denegar()")
    def espera(self):
        print("[STUB] espera()")
    def desconectar(self):
        print("[STUB] desconectar()")


# Al fondo de servo_control.py, reemplaza la creación de la instancia:
def _registrar_salida_bd():
    """Registra una salida anónima (botón) en la base de datos."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from database import conectar
        conn = conectar()
        conn.execute("""
            INSERT INTO registro_acceso
                (usuario_id, nombre, apellido_paterno, apellido_materno,
                 numero_cuenta, rol, tipo_evento, detalle)
            VALUES (NULL, 'Salida', 'Manual', '', NULL,
                    'estudiante', 'salida', 'Botón de salida presionado')
        """)
        conn.commit()
        conn.close()
        print("[HW] Salida registrada en BD")
    except Exception as e:
        print(f"[HW] Error registrando salida: {e}")

if _GPIO_OK:
    try:
        servo = _ControladorHW(on_salida_manual=_registrar_salida_bd)
    except Exception as e:
        print(f"[HW] Error al inicializar GPIO: {e}")
        servo = _ServoStub()
else:
    servo = _ServoStub()