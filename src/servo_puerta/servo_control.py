"""
servo_control.py  —  Raspberry Pi 5  (reemplaza Arduino + comunicación serial)
================================================================================
Controla directamente desde los GPIO de la Raspberry Pi:
  • Servomotor  → GPIO 18  (PWM hardware, pin físico 12)
  • LED VERDE   → GPIO 23  (pin físico 16)
  • LED ROJO    → GPIO 24  (pin físico 18)
  • Buzzer      → GPIO 25  (pin físico 22)
  • Botón man.  → GPIO 17  (pin físico 11)  [INPUT con pull-up interno]

Diagrama de pines Raspberry Pi 5:
  ┌─────────────────────────────────┐
  │  Pin 11 (GPIO17) ← Botón manual │
  │  Pin 12 (GPIO18) → Servo PWM    │
  │  Pin 16 (GPIO23) → LED Verde    │
  │  Pin 18 (GPIO24) → LED Rojo     │
  │  Pin 22 (GPIO25) → Buzzer       │
  │  Pin 6  (GND)    → GND común    │
  └─────────────────────────────────┘

Mismos comportamientos que el Arduino:
  • acceso  → LED verde ON, servo abre, buzzer tono corto 200ms,
              cierre automático a los 4 s, LED verde OFF
  • denegado → LED rojo parpadea 3 s, servo cierra, triple beep buzzer
  • espera  → todo apagado, servo cierra
  • botón   → igual que acceso (apertura manual)

Dependencias:
  pip install RPi.GPIO        (ya viene en Raspberry Pi OS)
  pip install pigpio          (alternativa para PWM más suave en el servo)

NOTA: Si no hay GPIO disponible (PC de desarrollo), el módulo carga un
      _ServoStub silencioso para no romper la interfaz.
"""

import time
import threading

# ── Intentar importar GPIO ────────────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    _GPIO_OK = True
except ImportError:
    _GPIO_OK = False
    print("[HW] RPi.GPIO no disponible — modo simulado activado")


# ── Configuración de pines (BCM) ──────────────────────────────────────────────
PIN_SERVO     = 18   # PWM hardware (GPIO18 / pin físico 12)
PIN_LED_VERDE = 23
PIN_LED_ROJO  = 24
PIN_BUZZER    = 25
PIN_BOTON     = 17   # INPUT_PULLUP

# ── Ángulos del servo ─────────────────────────────────────────────────────────
SERVO_ABIERTO  = 90   # grados
SERVO_CERRADO  =  0   # grados

# Conversión de ángulo → duty cycle para PWM a 50 Hz
# duty = 2.5 + (angulo / 180) * 10   →  0°=2.5%  90°=7.5%  180°=12.5%
def _angulo_a_duty(grados: float) -> float:
    return 2.5 + (grados / 180.0) * 10.0


# ── Tiempos (segundos) ────────────────────────────────────────────────────────
T_PUERTA_ABIERTA = 4.0
T_LED_DENEGADO   = 3.0
T_PARPADEO       = 0.3
T_BUZZER_ACCESO  = 0.2
T_BEEP           = 0.3    # duración de cada beep del triple beep
DEBOUNCE_S       = 0.05


# ══════════════════════════════════════════════════════════════════════════════
class _ControladorHW:
    """
    Gestiona el hardware GPIO en hilos no bloqueantes.
    Expone la misma API que el stub: abrir(), denegar(), espera(), desconectar()
    """

    def __init__(self):
        self._lock          = threading.Lock()
        self._servo_pwm     = None
        self._puerta_timer  = None
        self._denegado_timer = None
        self._beep_thread   = None
        self._boton_thread  = None
        self._running       = True

        self._setup_gpio()
        self._iniciar_monitor_boton()

    # ── Inicialización GPIO ───────────────────────────────────────────────────
    def _setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(PIN_LED_VERDE, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(PIN_LED_ROJO,  GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(PIN_BUZZER,    GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(PIN_BOTON,     GPIO.IN,  pull_up_down=GPIO.PUD_UP)

        # Servo con PWM a 50 Hz
        GPIO.setup(PIN_SERVO, GPIO.OUT)
        self._servo_pwm = GPIO.PWM(PIN_SERVO, 50)
        self._servo_pwm.start(_angulo_a_duty(SERVO_CERRADO))

        print("[HW] GPIO inicializado correctamente")
        print(f"[HW] Servo GPIO{PIN_SERVO} | LED Verde GPIO{PIN_LED_VERDE} "
              f"| LED Rojo GPIO{PIN_LED_ROJO} | Buzzer GPIO{PIN_BUZZER} "
              f"| Botón GPIO{PIN_BOTON}")

    # ── Control del servo ─────────────────────────────────────────────────────
    def _mover_servo(self, grados: float):
        if self._servo_pwm:
            self._servo_pwm.ChangeDutyCycle(_angulo_a_duty(grados))
            time.sleep(0.4)   # dar tiempo al servo para moverse
            # Detener la señal PWM para evitar vibración (servo pasivo)
            self._servo_pwm.ChangeDutyCycle(0)

    # ── Buzzer ────────────────────────────────────────────────────────────────
    def _beep_acceso(self):
        """Un beep corto de 200 ms para acceso permitido."""
        GPIO.output(PIN_BUZZER, GPIO.HIGH)
        time.sleep(T_BUZZER_ACCESO)
        GPIO.output(PIN_BUZZER, GPIO.LOW)

    def _triple_beep(self):
        """Triple beep no bloqueante para acceso denegado (en hilo aparte)."""
        def _run():
            for _ in range(3):
                if not self._running:
                    break
                GPIO.output(PIN_BUZZER, GPIO.HIGH)
                time.sleep(T_BEEP)
                GPIO.output(PIN_BUZZER, GPIO.LOW)
                time.sleep(T_BEEP)
        self._beep_thread = threading.Thread(target=_run, daemon=True)
        self._beep_thread.start()

    # ── Parpadeo LED rojo ─────────────────────────────────────────────────────
    def _parpadeo_rojo(self):
        """Parpadea el LED rojo durante T_LED_DENEGADO segundos en hilo aparte."""
        def _run():
            t_fin = time.time() + T_LED_DENEGADO
            while time.time() < t_fin and self._running:
                GPIO.output(PIN_LED_ROJO, GPIO.HIGH)
                time.sleep(T_PARPADEO)
                GPIO.output(PIN_LED_ROJO, GPIO.LOW)
                time.sleep(T_PARPADEO)
            GPIO.output(PIN_LED_ROJO, GPIO.LOW)
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ── Cierre automático ─────────────────────────────────────────────────────
    def _programar_cierre(self):
        """Cierra la puerta automáticamente después de T_PUERTA_ABIERTA s."""
        if self._puerta_timer:
            self._puerta_timer.cancel()
        self._puerta_timer = threading.Timer(T_PUERTA_ABIERTA, self._cerrar_puerta)
        self._puerta_timer.daemon = True
        self._puerta_timer.start()

    def _cerrar_puerta(self):
        with self._lock:
            self._mover_servo(SERVO_CERRADO)
            GPIO.output(PIN_LED_VERDE, GPIO.LOW)
        print("[HW] Puerta cerrada (auto)")

    # ── Botón manual ──────────────────────────────────────────────────────────
    def _iniciar_monitor_boton(self):
        """Hilo que monitorea el botón con anti-rebote por software."""
        def _monitor():
            estado_anterior = GPIO.HIGH
            while self._running:
                lectura = GPIO.input(PIN_BOTON)
                if lectura == GPIO.LOW and estado_anterior == GPIO.HIGH:
                    time.sleep(DEBOUNCE_S)   # anti-rebote
                    if GPIO.input(PIN_BOTON) == GPIO.LOW:
                        print("[HW] Botón presionado — apertura manual")
                        self.abrir("Manual")
                estado_anterior = lectura
                time.sleep(0.02)

        self._boton_thread = threading.Thread(target=_monitor, daemon=True)
        self._boton_thread.start()

    # ══════════════════════════════════════════════════════════════════════════
    #  API PÚBLICA  (misma que _ServoStub en interfaz.py)
    # ══════════════════════════════════════════════════════════════════════════

    def abrir(self, nombre: str = ""):
        """Acceso PERMITIDO: LED verde, servo abre, beep corto, cierre en 4 s."""
        print(f"[HW] ACCESO PERMITIDO — {nombre}")
        with self._lock:
            # Apagar cualquier estado de denegado activo
            GPIO.output(PIN_LED_ROJO, GPIO.LOW)

            GPIO.output(PIN_LED_VERDE, GPIO.HIGH)
            self._mover_servo(SERVO_ABIERTO)

        # Beep en hilo para no bloquear
        threading.Thread(target=self._beep_acceso, daemon=True).start()

        # Cierre automático
        self._programar_cierre()

    def denegar(self):
        """Acceso DENEGADO: LED rojo parpadea 3 s, servo cierra, triple beep."""
        print("[HW] ACCESO DENEGADO")
        with self._lock:
            GPIO.output(PIN_LED_VERDE, GPIO.LOW)
            self._mover_servo(SERVO_CERRADO)

        # Parpadeo y beeps en hilos separados (no bloqueantes)
        self._parpadeo_rojo()
        self._triple_beep()

    def espera(self):
        """MODO ESPERA: todo apagado, servo cerrado."""
        print("[HW] MODO ESPERA")
        if self._puerta_timer:
            self._puerta_timer.cancel()
        with self._lock:
            GPIO.output(PIN_LED_VERDE, GPIO.LOW)
            GPIO.output(PIN_LED_ROJO,  GPIO.LOW)
            GPIO.output(PIN_BUZZER,    GPIO.LOW)
            self._mover_servo(SERVO_CERRADO)

    def desconectar(self):
        """Limpieza al cerrar la aplicación."""
        print("[HW] Desconectando GPIO...")
        self._running = False
        if self._puerta_timer:
            self._puerta_timer.cancel()
        if self._servo_pwm:
            self._servo_pwm.stop()
        GPIO.output(PIN_LED_VERDE, GPIO.LOW)
        GPIO.output(PIN_LED_ROJO,  GPIO.LOW)
        GPIO.output(PIN_BUZZER,    GPIO.LOW)
        GPIO.cleanup()
        print("[HW] GPIO liberado.")


# ══════════════════════════════════════════════════════════════════════════════
class _ServoStub:
    """Fallback silencioso cuando no hay GPIO disponible (PC de desarrollo)."""
    def abrir(self, nombre: str = ""):
        print(f"[STUB] abrir() — {nombre}")
    def denegar(self):
        print("[STUB] denegar()")
    def espera(self):
        print("[STUB] espera()")
    def desconectar(self):
        print("[STUB] desconectar()")


# ── Instancia global exportada (interfaz.py hace: from servo_control import servo)
if _GPIO_OK:
    try:
        servo = _ControladorHW()
    except Exception as e:
        print(f"[HW] Error al inicializar GPIO: {e}")
        servo = _ServoStub()
else:
    servo = _ServoStub()