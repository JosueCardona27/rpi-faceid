"""
ultrasonico.py  —  Raspberry Pi 5  (HC-SR04 + control de cámara)
================================================================================
Activa y desactiva la cámara según la distancia detectada por el HC-SR04.

  • Si detecta persona a menos de 70 cm → enciende cámara
  • Si no detecta persona por 3 segundos → apaga cámara

PINES (BCM):
  • TRIG → GPIO 20  (pin físico 38)
  • ECHO → GPIO 21  (pin físico 40)  ← con divisor de voltaje obligatorio

DIVISOR DE VOLTAJE para ECHO (5V → 3.3V):
  ECHO sensor → Resistencia 1kΩ → GPIO21
                                 ↓
                           Resistencia 2kΩ
                                 ↓
                               GND

Dependencias:
  pip install lgpio --break-system-packages
"""

import time
import threading
import lgpio

# ── Pines ─────────────────────────────────────────────────────────────────────
PIN_TRIG = 20   # pin físico 38
PIN_ECHO = 21   # pin físico 40

# ── Configuración ─────────────────────────────────────────────────────────────
DISTANCIA_ACTIVACION_CM = 70     # distancia máxima para activar cámara
TIEMPO_SIN_PERSONA_S    = 3.0    # segundos sin detectar antes de apagar cámara
INTERVALO_MEDICION_S    = 0.2    # cada cuánto mide (5 veces por segundo)
TIMEOUT_ECHO_S          = 0.03   # timeout si el echo no llega (objeto muy lejos)


# ══════════════════════════════════════════════════════════════════════════════
class SensorUltrasonico:
    """
    Mide distancia con HC-SR04 y notifica a interfaz.py
    cuándo encender o apagar la cámara.

    Uso desde interfaz.py:
        from ultrasonico import SensorUltrasonico
        self.sensor = SensorUltrasonico(
            on_persona=self._start_cam,
            on_sin_persona=self._stop_cam
        )
        self.sensor.iniciar()

        # Al cerrar la app:
        self.sensor.detener()
    """

    def __init__(self, on_persona=None, on_sin_persona=None):
        """
        on_persona     : función a llamar cuando se detecta persona (enciende cámara)
        on_sin_persona : función a llamar cuando la persona se aleja (apaga cámara)
        """
        self._on_persona      = on_persona     or (lambda: None)
        self._on_sin_persona  = on_sin_persona or (lambda: None)

        self._handle          = None
        self._running         = False
        self._hilo            = None
        self._camara_activa   = False
        self._t_ultima_detec  = 0.0

        self._setup_gpio()

    # ── GPIO ──────────────────────────────────────────────────────────────────
    def _setup_gpio(self):
        try:
            self._handle = lgpio.gpiochip_open(4)   # gpiochip4 en Pi 5
        except Exception:
            self._handle = lgpio.gpiochip_open(0)

        lgpio.gpio_claim_output(self._handle, PIN_TRIG, 0)
        lgpio.gpio_claim_input(self._handle,  PIN_ECHO)

        print(f"[SENSOR] HC-SR04 listo — TRIG GPIO{PIN_TRIG} | ECHO GPIO{PIN_ECHO}")
        print(f"[SENSOR] Distancia de activación: {DISTANCIA_ACTIVACION_CM} cm")

    # ── Medición ──────────────────────────────────────────────────────────────
    def _medir_distancia(self) -> float:
        """
        Envía pulso TRIG y mide el tiempo de respuesta ECHO.
        Devuelve distancia en cm, o -1 si hay timeout.
        """
        # Asegurarse que TRIG está en LOW
        lgpio.gpio_write(self._handle, PIN_TRIG, 0)
        time.sleep(0.000002)   # 2 µs

        # Pulso TRIG de 10 µs
        lgpio.gpio_write(self._handle, PIN_TRIG, 1)
        time.sleep(0.00001)    # 10 µs
        lgpio.gpio_write(self._handle, PIN_TRIG, 0)

        # Esperar flanco de subida del ECHO
        t_inicio = time.time()
        while lgpio.gpio_read(self._handle, PIN_ECHO) == 0:
            if time.time() - t_inicio > TIMEOUT_ECHO_S:
                return -1   # timeout

        # Medir duración del pulso ECHO
        t_subida = time.time()
        while lgpio.gpio_read(self._handle, PIN_ECHO) == 1:
            if time.time() - t_subida > TIMEOUT_ECHO_S:
                return -1   # timeout

        t_bajada = time.time()

        # Distancia = (tiempo * velocidad_sonido) / 2
        # velocidad sonido = 34300 cm/s
        duracion   = t_bajada - t_subida
        distancia  = (duracion * 34300) / 2
        return round(distancia, 1)

    # ── Loop principal ────────────────────────────────────────────────────────
    def _loop(self):
        print("[SENSOR] Monitoreo iniciado")
        while self._running:
            distancia = self._medir_distancia()

            if distancia > 0:
                print(f"[SENSOR] Distancia: {distancia} cm", end="\r")

            persona_cerca = (0 < distancia <= DISTANCIA_ACTIVACION_CM)

            if persona_cerca:
                self._t_ultima_detec = time.time()

                if not self._camara_activa:
                    self._camara_activa = True
                    print(f"\n[SENSOR] Persona detectada a {distancia} cm — encendiendo cámara")
                    try:
                        self._on_persona()
                    except Exception as e:
                        print(f"[SENSOR] Error al encender cámara: {e}")

            else:
                tiempo_sin_persona = time.time() - self._t_ultima_detec

                if self._camara_activa and tiempo_sin_persona >= TIEMPO_SIN_PERSONA_S:
                    self._camara_activa = False
                    print(f"\n[SENSOR] Sin persona por {TIEMPO_SIN_PERSONA_S}s — apagando cámara")
                    try:
                        self._on_sin_persona()
                    except Exception as e:
                        print(f"[SENSOR] Error al apagar cámara: {e}")

            time.sleep(INTERVALO_MEDICION_S)

    # ── API pública ───────────────────────────────────────────────────────────
    def iniciar(self):
        """Inicia el monitoreo en un hilo separado."""
        self._running        = True
        self._t_ultima_detec = time.time()
        self._hilo = threading.Thread(target=self._loop, daemon=True)
        self._hilo.start()

    def detener(self):
        """Detiene el monitoreo y libera el GPIO."""
        print("\n[SENSOR] Deteniendo sensor...")
        self._running = False
        if self._hilo:
            self._hilo.join(timeout=2)
        try:
            lgpio.gpio_write(self._handle, PIN_TRIG, 0)
            lgpio.gpiochip_close(self._handle)
        except Exception:
            pass
        print("[SENSOR] GPIO sensor liberado.")

    def distancia_actual(self) -> float:
        """Devuelve una medición instantánea (útil para pruebas)."""
        return self._medir_distancia()

    @property
    def camara_activa(self) -> bool:
        return self._camara_activa


# ══════════════════════════════════════════════════════════════════════════════
#  PRUEBA INDEPENDIENTE
#  Corre este archivo directamente para verificar que el sensor funciona:
#  python3 ultrasonico.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== Prueba HC-SR04 ===")
    print("Presiona Ctrl+C para salir\n")

    def camara_on():
        print(">>> CÁMARA ENCENDIDA")

    def camara_off():
        print(">>> CÁMARA APAGADA")

    sensor = SensorUltrasonico(on_persona=camara_on, on_sin_persona=camara_off)
    sensor.iniciar()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sensor.detener()
        print("\nPrueba finalizada.")