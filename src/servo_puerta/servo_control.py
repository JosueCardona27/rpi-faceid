"""
servo_control.py
================
Controla el Arduino Nano via puerto serial (USB).

Comandos que entiende el Arduino (control_puerta.ino):
  acceso:<nombre>  ->  abre servo 90 + LED verde 4 segundos
  denegado         ->  parpadeo LED rojo 3 segundos
  espera           ->  apaga todo, cierra servo

Uso desde interfaz.py:
  from servo_puerta.servo_control import servo
  servo.abrir("Juan Perez")
  servo.denegar()
  servo.espera()
  servo.desconectar()
"""

import threading
import time

try:
    import serial
    import serial.tools.list_ports
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False
    print("[SERVO] pyserial no instalado — ejecuta:  pip install pyserial")


BAUDRATE       = 9600
TIMEOUT_CON    = 2
RECONECTAR_SEG = 5

_KEYWORDS = ("Arduino", "CH340", "CP210", "ttyUSB", "ttyACM", "usbserial", "USB Serial")


def _detectar_puerto():
    if not _SERIAL_OK:
        return None

    for p in serial.tools.list_ports.comports():
        desc = f"{p.description} {p.manufacturer or ''}"
        if any(k.lower() in desc.lower() for k in _KEYWORDS):
            print(f"[SERVO] Arduino detectado en {p.device}  ({p.description})")
            return p.device
        if p.device.startswith("/dev/ttyACM") or \
           p.device.startswith("/dev/ttyUSB"):
            print(f"[SERVO] Puerto probable detectado: {p.device}")
            return p.device

    print("[SERVO] No se encontro ningun Arduino. Puertos disponibles:")
    for p in serial.tools.list_ports.comports():
        print(f"         {p.device}  -  {p.description}")
    return None


class _ServoArduino:

    def __init__(self, port=None):
        self._port   = port
        self._ser    = None
        self._lock   = threading.Lock()
        self._activo = True
        threading.Thread(target=self._hilo_conexion,
                         daemon=True, name="servo-conexion").start()

    def _hilo_conexion(self):
        while self._activo:
            if self._conectado():
                time.sleep(1)
                continue
            puerto = self._port or _detectar_puerto()
            if puerto:
                self._abrir_puerto(puerto)
            else:
                print(f"[SERVO] Reintentando en {RECONECTAR_SEG}s...")
            time.sleep(RECONECTAR_SEG)

    def _abrir_puerto(self, puerto):
        try:
            ser = serial.Serial(puerto, BAUDRATE, timeout=TIMEOUT_CON)
            print(f"[SERVO] Conectando a {puerto}, esperando reinicio del Arduino...")
            time.sleep(2)
            with self._lock:
                self._ser = ser
            print(f"[SERVO] Conectado en {puerto} a {BAUDRATE} baud")
        except Exception as e:
            print(f"[SERVO] Error al abrir {puerto}: {e}")

    def _conectado(self):
        with self._lock:
            return self._ser is not None and self._ser.is_open

    def _enviar(self, cmd):
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                print(f"[SERVO] Sin conexion — comando ignorado: '{cmd}'")
                return
            try:
                self._ser.write((cmd + "\n").encode("utf-8"))
                self._ser.flush()
                print(f"[SERVO] -> Enviado: '{cmd}'")
            except Exception as e:
                print(f"[SERVO] Error al enviar '{cmd}': {e}")
                try: self._ser.close()
                except: pass
                self._ser = None

    def abrir(self, nombre=""):
        nombre_limpio = nombre.strip()[:40]
        if nombre_limpio:
            self._enviar(f"acceso:{nombre_limpio}")
        else:
            self._enviar("abrir")

    def denegar(self):
        self._enviar("denegado")

    def espera(self):
        self._enviar("espera")

    def desconectar(self):
        self._activo = False
        self.espera()
        time.sleep(0.4)
        with self._lock:
            if self._ser and self._ser.is_open:
                try: self._ser.close()
                except: pass
                self._ser = None
        print("[SERVO] Desconectado correctamente.")


if _SERIAL_OK:
    servo = _ServoArduino()
    # Para forzar puerto manual descomenta la linea de tu SO:
    # servo = _ServoArduino(port="COM3")            # Windows
    # servo = _ServoArduino(port="/dev/ttyUSB0")    # Linux / Raspberry Pi
else:
    class _Stub:
        def abrir(self, nombre=""): pass
        def denegar(self):          pass
        def espera(self):           pass
        def desconectar(self):      pass
    servo = _Stub()