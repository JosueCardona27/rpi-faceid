"""
interfaz.py
===========
Interfaz grafica del sistema de acceso facial.
Pantalla tactil 7" 1024x600.

Formulario de registro:
  - Todos:          nombre, apellidos, numero de cuenta (8 digitos)
  - Admin/Maestro:  correo + contrasena (minimo 6 caracteres)
  - Estudiante:     grado (1-20) + grupo (A-Z)

v5.5  |  4 pasos  |  MobileFaceNet 512 dims
"""

import tkinter as tk
from tkinter import font as tkfont
import cv2
import numpy as np
import threading
import hashlib
import time
from PIL import Image, ImageTk

USAR_PICAM = False
try:
    from picamera2 import Picamera2
    _test = Picamera2(); _test.close(); del _test
    USAR_PICAM = True
    print("[CAM] Picamera2 detectada")
except Exception:
    print("[CAM] Modo webcam OpenCV")

from face_engine import (extraer_caracteristicas, dibujar_overlay,
                          TIPO_FRONTAL, TIPO_PERFIL_D, TIPO_PERFIL_I,
                          detectar_oclusion)
import face_engine as _fe
from database   import (registrar_usuario, guardar_vectores_por_angulo,
                         guardar_vector_unico, reconocer_persona,
                         eliminar_persona, verificar_duplicado_facial,
                         validar_numero_cuenta, validar_correo,
                         validar_contrasena, validar_grado,
                         validar_grupo, ROLES_VALIDOS,
                         registrar_acceso, conectar)

try:
    from servo_puerta.servo_control import servo
except Exception as e:
    print(f"[SERVO] Error al importar servo_control: {e}")
    class _ServoStub:
        def abrir(self, nombre=""): pass
        def denegar(self):          pass
        def espera(self):           pass
        def desconectar(self):      pass
    servo = _ServoStub()
    print("[SERVO] Modulo no disponible — continuando sin servo.")

try:
    from ultrasonico import SensorUltrasonico
    _SENSOR_OK = True
except Exception as e:
    print(f"[SENSOR] Error al importar ultrasonico: {e}")
    _SENSOR_OK = False

BG      = "#F5F0E8"   # beige institucional (fondo general)
PANEL   = "#FFFFFF"   # blanco puro (paneles/cards)
CARD    = "#EAE5D8"   # beige claro (fondos de campos)
ACCENT  = "#006644"   # verde loro institucional UdeC
ACCENT2 = "#008855"   # verde loro medio
SUCCESS = "#006644"   # verde loro (éxito)
DANGER  = "#C1121F"   # rojo institucional
WARNING = "#E07A00"   # naranja advertencia
TEXT    = "#1A1A2E"   # texto oscuro casi negro
SUBTEXT = "#5C6170"   # gris medio para subtextos
BORDER  = "#C8C2B2"   # borde beige gris

# Paleta institucional UdeC
NAVY_LN = "#1B2A4A"   # azul marino institucional UdeC
TEAL_LN = "#007A55"   # verde azulado UdeC
GOLD_LN = "#B5860D"   # dorado institucional UdeC

COLOR_ROL = {"estudiante": NAVY_LN, "maestro": TEAL_LN, "admin": ACCENT}

W, H    = 600, 1024
PANEL_H = 400   # altura del panel de formulario/resultados (parte superior)
CAM_H_V = H - PANEL_H  # altura de la zona de cámara (parte inferior)
# Compatibilidad con referencias antiguas
PANEL_W = W
CAM_W   = W

HAAR_PATH = None

GRADOS = [str(i) for i in range(1, 11)]
GRUPOS = [chr(i) for i in range(ord('A'), ord('G'))]

# ── Pasos de registro ─────────────────────────────────────────────────────────
PASOS_REGISTRO = [
    (0, "●", "Mira directo a la camara",
     "FRENTE",    12.0, "frontal", TIPO_FRONTAL,  "Mira directo a la camara"),
    (1, "◀", "Gira tu cabeza a la IZQUIERDA",
     "IZQUIERDA", 10.0, "perfil",  TIPO_PERFIL_D, "Gira mas a tu izquierda"),
    (2, "▶", "Gira tu cabeza a la DERECHA",
     "DERECHA",   10.0, "perfil",  TIPO_PERFIL_I, "Gira mas a tu derecha"),
    (3, "●", "Vuelve al frente",
     "FRENTE",    12.0, "frontal", TIPO_FRONTAL,  "Mira directo a la camara"),
]
N_PASOS           = len(PASOS_REGISTRO)
TIEMPO_ESCANEO    = sum(p[4] for p in PASOS_REGISTRO)
MAX_MUESTRAS_PASO = 50
MUESTRAS_MIN_PASO = 10


def _imgtk(frame, max_w, max_h):
    h0, w0 = frame.shape[:2]
    r  = min(max_w / w0, max_h / h0)
    fr = cv2.resize(frame, (int(w0*r), int(h0*r)))
    return ImageTk.PhotoImage(
        image=Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))


def _draw_multiface_banner(frame):
    """
    Dibuja un banner centrado grande sobre el frame cuando hay varios
    rostros detectados. No dibuja ningun bbox para evitar oscilaciones
    visuales entre las caras detectadas.
    """
    h_img, w_img = frame.shape[:2]
    overlay = frame.copy()

    # Fondo oscuro semitransparente sobre todo el frame
    cv2.rectangle(overlay, (0, 0), (w_img, h_img), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Panel central
    font  = cv2.FONT_HERSHEY_DUPLEX
    msg1  = "NO SE PUEDE ESCANEAR"
    msg2  = "Se detectaron varios rostros"
    msg3  = "Solo una persona a la vez"

    s1, s2, s3 = 0.9, 0.65, 0.55
    th1, th2, th3 = 2, 1, 1

    (w1, h1), _ = cv2.getTextSize(msg1, font, s1, th1)
    (w2, h2), _ = cv2.getTextSize(msg2, font, s2, th2)
    (w3, h3), _ = cv2.getTextSize(msg3, font, s3, th3)

    total_h = h1 + 14 + h2 + 8 + h3 + 30
    box_w   = max(w1, w2, w3) + 60
    box_h   = total_h
    bx1     = (w_img - box_w) // 2
    by1     = (h_img - box_h) // 2
    bx2     = bx1 + box_w
    by2     = by1 + box_h

    # Caja con borde naranja
    bg = frame.copy()
    cv2.rectangle(bg, (bx1, by1), (bx2, by2), (10, 10, 25), -1)
    cv2.addWeighted(bg, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 140, 255), 3, cv2.LINE_AA)

    # Texto centrado
    y = by1 + 15 + h1
    cv2.putText(frame, msg1, ((w_img - w1) // 2, y),
                font, s1, (0, 0, 0), th1 + 3, cv2.LINE_AA)
    cv2.putText(frame, msg1, ((w_img - w1) // 2, y),
                font, s1, (0, 140, 255), th1, cv2.LINE_AA)

    y += h2 + 14
    cv2.putText(frame, msg2, ((w_img - w2) // 2, y),
                font, s2, (255, 255, 255), th2, cv2.LINE_AA)

    y += h3 + 8
    cv2.putText(frame, msg3, ((w_img - w3) // 2, y),
                font, s3, (200, 200, 200), th3, cv2.LINE_AA)


    return frame


def _dibujar_overlay_estetico(frame, coords, color=(0, 212, 255), texto="", tipo=None):
    """
    Overlay visual más limpio para pantalla vertical táctil.
    - No cambia la detección ni el vector facial.
    - Solo reduce y estiliza el recuadro dibujado.
    - Mantiene fondo negro translúcido detrás del texto.
    """
    if coords is None:
        return frame
    try:
        x, y, w, h = [int(v) for v in coords[:4]]
    except Exception:
        return frame

    h_img, w_img = frame.shape[:2]
    # Recuadro visual más pequeño que el bbox real para que no invada media pantalla.
    pad_x = int(w * 0.13)
    pad_y_top = int(h * 0.12)
    pad_y_bot = int(h * 0.08)
    x1 = max(4, x + pad_x)
    y1 = max(4, y + pad_y_top)
    x2 = min(w_img - 5, x + w - pad_x)
    y2 = min(h_img - 5, y + h - pad_y_bot)
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = max(4, x), max(4, y), min(w_img-5, x+w), min(h_img-5, y+h)

    # Suavizar un poco el color para que se vea institucional, no tan agresivo.
    b, g, r = [int(c) for c in color]
    glow = (min(255, b + 20), min(255, g + 20), min(255, r + 20))
    main = (b, g, r)
    shadow = (5, 8, 15)

    # Sombra sutil en esquinas.
    L = max(28, min(x2 - x1, y2 - y1) // 4)
    thick_shadow = 7
    thick = 4
    def corners(col, thickness, offset=0):
        xx1, yy1, xx2, yy2 = x1+offset, y1+offset, x2+offset, y2+offset
        cv2.line(frame, (xx1, yy1), (xx1 + L, yy1), col, thickness, cv2.LINE_AA)
        cv2.line(frame, (xx1, yy1), (xx1, yy1 + L), col, thickness, cv2.LINE_AA)
        cv2.line(frame, (xx2, yy1), (xx2 - L, yy1), col, thickness, cv2.LINE_AA)
        cv2.line(frame, (xx2, yy1), (xx2, yy1 + L), col, thickness, cv2.LINE_AA)
        cv2.line(frame, (xx1, yy2), (xx1 + L, yy2), col, thickness, cv2.LINE_AA)
        cv2.line(frame, (xx1, yy2), (xx1, yy2 - L), col, thickness, cv2.LINE_AA)
        cv2.line(frame, (xx2, yy2), (xx2 - L, yy2), col, thickness, cv2.LINE_AA)
        cv2.line(frame, (xx2, yy2), (xx2, yy2 - L), col, thickness, cv2.LINE_AA)
    corners(shadow, thick_shadow, offset=2)
    corners(glow, thick + 2, offset=0)
    corners(main, thick, offset=0)

    # Pequeñas líneas laterales semitransparentes para guía sin cerrar todo el cuadro.
    overlay = frame.copy()
    mid_y = (y1 + y2) // 2
    cv2.line(overlay, (x1, mid_y), (x1, mid_y), main, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX
    titulo = texto.strip() if texto else "Detectado"
    subtitulo = tipo.upper() if tipo else ""
    titulo = titulo.split("\n")[0][:22]

    # Caja superior negra translúcida, como pediste, pero menos grande.
    scale_t = 0.78 if len(titulo) <= 14 else 0.66
    th = 2
    (tw, thh), _ = cv2.getTextSize(titulo, font, scale_t, th)
    label_w = min(w_img - 16, max(tw + 34, 170))
    label_h = 50 if subtitulo else 42
    lx1 = max(8, min(w_img - label_w - 8, (x1 + x2 - label_w) // 2))
    ly2 = max(50, y1 - 10)
    ly1 = max(6, ly2 - label_h)

    box = frame.copy()
    cv2.rectangle(box, (lx1, ly1), (lx1 + label_w, ly1 + label_h), (0, 0, 0), -1)
    cv2.addWeighted(box, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (lx1, ly1), (lx1 + label_w, ly1 + label_h), main, 2, cv2.LINE_AA)

    tx = lx1 + (label_w - tw) // 2
    ty = ly1 + 28
    cv2.putText(frame, titulo, (tx, ty), font, scale_t, shadow, th + 3, cv2.LINE_AA)
    cv2.putText(frame, titulo, (tx, ty), font, scale_t, main, th, cv2.LINE_AA)

    if subtitulo:
        scale_s = 0.52
        (sw, sh), _ = cv2.getTextSize(subtitulo, font, scale_s, 1)
        cv2.putText(frame, subtitulo, (lx1 + (label_w - sw)//2, ly1 + label_h - 10),
                    font, scale_s, (235, 245, 255), 1, cv2.LINE_AA)
    return frame


# ── Helpers de dibujo ─────────────────────────────────────────────────────────
def _round_rect(cv, x1, y1, x2, y2, r=14,
                fill=None, outline=None, width=1, tags=""):
    """Rectángulo con esquinas redondeadas dibujado en un Canvas."""
    fill    = fill    or PANEL
    outline = outline or BORDER
    cv.create_arc(x1,     y1,     x1+2*r, y1+2*r, start=90,  extent=90,
                  fill=fill, outline=fill, tags=tags)
    cv.create_arc(x2-2*r, y1,     x2,     y1+2*r, start=0,   extent=90,
                  fill=fill, outline=fill, tags=tags)
    cv.create_arc(x1,     y2-2*r, x1+2*r, y2,     start=180, extent=90,
                  fill=fill, outline=fill, tags=tags)
    cv.create_arc(x2-2*r, y2-2*r, x2,     y2,     start=270, extent=90,
                  fill=fill, outline=fill, tags=tags)
    cv.create_rectangle(x1+r, y1,   x2-r, y2,   fill=fill, outline="", tags=tags)
    cv.create_rectangle(x1,   y1+r, x2,   y2-r, fill=fill, outline="", tags=tags)
    if width > 0:
        cv.create_arc(x1,     y1,     x1+2*r, y1+2*r, start=90,  extent=90,
                      style="arc", outline=outline, width=width, tags=tags)
        cv.create_arc(x2-2*r, y1,     x2,     y1+2*r, start=0,   extent=90,
                      style="arc", outline=outline, width=width, tags=tags)
        cv.create_arc(x1,     y2-2*r, x1+2*r, y2,     start=180, extent=90,
                      style="arc", outline=outline, width=width, tags=tags)
        cv.create_arc(x2-2*r, y2-2*r, x2,     y2,     start=270, extent=90,
                      style="arc", outline=outline, width=width, tags=tags)
        cv.create_line(x1+r, y1,   x2-r, y1,   fill=outline, width=width, tags=tags)
        cv.create_line(x1+r, y2,   x2-r, y2,   fill=outline, width=width, tags=tags)
        cv.create_line(x1,   y1+r, x1,   y2-r, fill=outline, width=width, tags=tags)
        cv.create_line(x2,   y1+r, x2,   y2-r, fill=outline, width=width, tags=tags)


def _rounded_btn(parent, text, cmd, width=284, height=40,
                 bg=None, fg="#FFFFFF", hover=None, font_size=9, bg_parent=None):
    """Botón completamente redondeado usando Canvas + Label."""
    bg        = bg        or ACCENT
    hover     = hover     or ACCENT2
    bg_parent = bg_parent or PANEL
    r = height // 2
    frame = tk.Frame(parent, bg=bg_parent, width=width, height=height)
    frame.pack_propagate(False)
    cv2_ = tk.Canvas(frame, width=width, height=height,
                     bg=bg_parent, highlightthickness=0)
    cv2_.place(x=0, y=0)

    def _draw(color):
        cv2_.delete("all")
        _round_rect(cv2_, 0, 0, width, height, r=r,
                    fill=color, outline=color, width=0)

    _draw(bg)
    lbl = tk.Label(frame, text=text,
                   font=("Segoe UI", font_size, "bold"),
                   fg=fg, bg=bg, cursor="hand2")
    lbl.place(relx=.5, rely=.5, anchor="center")

    def on_enter(e):  _draw(hover); lbl.config(bg=hover)
    def on_leave(e):  _draw(bg);    lbl.config(bg=bg)
    def on_click(e):  cmd()

    for w in (cv2_, lbl):
        w.bind("<Enter>",    on_enter)
        w.bind("<Leave>",    on_leave)
        w.bind("<Button-1>", on_click)
    return frame


def _lighten_color(hx):
    h = hx.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"#{min(255,r+40):02x}{min(255,g+40):02x}{min(255,b+40):02x}"


# ══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Acceso Facial")
        self.geometry(f"{W}x{H}+0+0")
        self.resizable(False, False)
        # Sin bordes solo en Raspberry Pi (pantalla embebida)
        import platform
        if "raspberry" in platform.node().lower() or "raspberrypi" in platform.node().lower():
            self.overrideredirect(True)
        self.configure(bg=BG)

        FONT = "Segoe UI"
        self.f_title  = tkfont.Font(family=FONT, size=16, weight="bold")
        self.f_sub    = tkfont.Font(family=FONT, size=9)
        self.f_btn    = tkfont.Font(family=FONT, size=10, weight="bold")
        self.f_label  = tkfont.Font(family=FONT, size=9)
        self.f_status = tkfont.Font(family=FONT, size=9,  weight="bold")
        self.f_zona   = tkfont.Font(family=FONT, size=8)

        self.picam2      = None
        self._cap        = None
        self.cam_running = False
        self.verificando = False

        self._frame_actual = None
        self._frame_lock   = threading.Lock()

        self._analisis = {"vector": None, "coords": None,
                          "frame_id": -1, "tipo": None,
                          "ocluido": False, "razon_oclusion": ""}
        self._analisis_lock  = threading.Lock()
        self._modo_deteccion = "auto"
        self._tipo_esperado  = None

        # Variables del formulario
        self.nombre_var  = tk.StringVar()
        self.ap_pat_var  = tk.StringVar()
        self.ap_mat_var  = tk.StringVar()
        self.cuenta_var  = tk.StringVar()
        self.correo_var  = tk.StringVar()    # solo admin/maestro
        self.pwd_var     = tk.StringVar()    # solo admin/maestro
        self.rol_var     = tk.StringVar(value="estudiante")
        self.grado_var   = tk.StringVar(value="1")
        self.grupo_var   = tk.StringVar(value="A")
        self.status_var  = tk.StringVar(value="Listo")

        # Usuario autenticado desde el login web
        self._usuario_login: dict | None = None

        self._ov_color = None
        self._ov_texto = ""
        self._modo_acceso = False
        self._t_acceso_ok = 0 
        self._ov_lock  = threading.Lock()

        # Pantalla inicial automática: al abrir el programa entra directo a ACCESO.
        # El menú sigue disponible desde el botón “☰ Menú” de la pantalla de acceso.
        self.after(100, self._show_acceso)

        # ── Sensor ultrasónico ────────────────────────────────────────────────
        self.sensor = None
        if _SENSOR_OK:
            try:
                self.sensor = SensorUltrasonico(
                    on_persona     = self._sensor_persona_detectada,
                    on_sin_persona = self._sensor_sin_persona,
                )
                self.sensor.iniciar()
                print("[SENSOR] Sensor ultrasónico activo")
            except Exception as e:
                print(f"[SENSOR] No se pudo iniciar el sensor: {e}")
                self.sensor = None

        self.sensores_ir = None
        try:
            from sensores_ir import SensoresIR
            self.sensores_ir = SensoresIR(
                on_entrada = self._on_entrada_sensor,
                on_salida  = self._on_salida_sensor,
            )
            self.sensores_ir.iniciar()
            print("[IR] Sensores FC-51 activos")
        except Exception as e:
            print(f"[IR] No se pudieron iniciar los sensores IR: {e}")
            self.sensores_ir = None

        # ── Conectar botón de salida con el contador ──────────────────────────
        servo._on_salida_manual = self._on_salida_boton

    @staticmethod
    def _lighten(hx):
        h = hx.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{min(255,r+40):02x}{min(255,g+40):02x}{min(255,b+40):02x}"

    def _clear(self):
        for w in self.winfo_children():
            w.destroy()

    def _stop_cam(self):
        self.cam_running = False
        time.sleep(0.25)
        if USAR_PICAM:
            if self.picam2:
                try: self.picam2.close()
                except: pass
                self.picam2 = None
        else:
            if self._cap:
                try: self._cap.release()
                except: pass
                self._cap = None

    def _volver(self):
        self._stop_cam()
        self._usuario_login = None
        self._modo_acceso   = False
        self._build_main()

    # ── Callbacks del sensor ultrasónico ─────────────────────────────────────
    def _sensor_persona_detectada(self):
        """El sensor detectó una persona — encender cámara si está en modo acceso."""
        if self._modo_acceso and not self.cam_running:
            print("[SENSOR] Persona detectada — encendiendo cámara")
            self.after(0, self._activar_camara_acceso)

    def _sensor_sin_persona(self):
        """La persona se alejó — apagar cámara y modo espera."""
        if self._modo_acceso and self.cam_running:
            print("[SENSOR] Sin persona — apagando cámara")
            self.after(0, self._desactivar_camara_acceso)

    def _activar_camara_acceso(self):
        """Enciende la cámara en la pantalla de acceso."""
        if self.cam_running:
            return
        try:
            self.cam_running = True
            self._start_cam()
            self._resetear_pantalla_acceso()
            threading.Thread(target=self._loop_camara,
                             args=(CAM_W, CAM_H_V), daemon=True).start()
            threading.Thread(target=self._loop_analisis,
                             daemon=True).start()
            threading.Thread(target=self._monitor_cara,
                             daemon=True).start()
            threading.Thread(target=self._guia_posicion,
                             daemon=True).start()
        except Exception as e:
            print(f"[SENSOR] Error al encender cámara: {e}")
            self.cam_running = False

    def _desactivar_camara_acceso(self):
        """Apaga la cámara y muestra pantalla de espera."""
        if not self.cam_running:
            return
        self._stop_cam()
        self.after(300, lambda: self.cam_label.configure(image=""))
        servo.espera()
        try:
            self.resultado_var.set("Esperando persona...")
            self.resultado_label.config(fg=ACCENT)
            self.candidato_var.set("")
            self.detalle_var.set("")
            self._set_sim_bar(0, BORDER)
            self._set_overlay(None, "")
            self.hdr_status_var.set("")
            self.hdr_nombre_var.set("")
            self._draw_pill(ACCENT2)
            self.posicion_var.set("Acércate para identificarte")
        except Exception:
            pass

    def _set_overlay(self, color, texto=""):
        with self._ov_lock:
            self._ov_color = color
            self._ov_texto = texto

    def _start_cam(self):
        if USAR_PICAM:
            self.picam2 = Picamera2()
            cfg = self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "BGR888"})
            self.picam2.configure(cfg)
            self.picam2.start()
            time.sleep(0.3)
        else:
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self._cap.set(cv2.CAP_PROP_FPS, 15)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self._cap.isOpened():
                raise RuntimeError("No se pudo abrir ninguna camara.")
            for _ in range(5):
                self._cap.read()

    def _leer_frame(self):
        try:
            if USAR_PICAM:
                return self.picam2.capture_array()
            else:
                ret, raw = self._cap.read()
                return raw if ret else None
        except:
            return None

    def _loop_camara(self, max_w, max_h):
        while self.cam_running:
            raw = self._leer_frame()
            if raw is None:
                time.sleep(0.03)
                continue

            frame = cv2.flip(raw, 1)
            with self._frame_lock:
                self._frame_actual = frame

            with self._analisis_lock:
                coords         = self._analisis["coords"]
                vector         = self._analisis["vector"]
                tipo           = self._analisis["tipo"]
                ocluido        = self._analisis.get("ocluido", False)
                razon_oclusion = self._analisis.get("razon_oclusion", "")

            with self._ov_lock:
                ov_color = self._ov_color
                ov_texto = self._ov_texto

            _msgs_oc = {
                "mascara":     "Descubre tu rostro",
                "gorra":       "Descubre tu rostro",
                "mano":        "Descubre tu rostro",
                "lentes":      "Descubre tu rostro",
                "obstruccion": "Descubre tu rostro",
            }

            vis = frame.copy()

            # Caso especial: varios rostros similares en camara.
            # No se dibuja ningun bbox para evitar oscilaciones.
            # Solo se muestra un banner grande en el centro.
            if _fe._multiple_faces and coords is None:
                vis = _draw_multiface_banner(vis)
            elif coords:
                # La oclusión tiene prioridad tanto en registro como en acceso.
                # Antes en modo acceso se ignoraba, por eso podía salir "Detectado"
                # aunque la boca estuviera cubierta.
                if ocluido:
                    msg = _msgs_oc.get(razon_oclusion, "Descubre tu rostro")
                    vis = dibujar_overlay(vis, coords, (255, 130, 0),
                                          msg, tipo=tipo)
                elif self._modo_acceso:
                    if ov_color:
                        c = ov_color
                    elif vector is not None and tipo == TIPO_FRONTAL:
                        c = (0, 212, 255)
                    elif vector is not None:
                        c = (255, 184, 48)
                    else:
                        c = (80, 80, 80)
                    t = ov_texto if ov_texto else ""
                    vis = dibujar_overlay(vis, coords, c, t, tipo=None)
                else:
                    c = ov_color if ov_color else (
                        (0, 212, 255) if vector is not None else (80, 80, 80))
                    t = ov_texto if ov_texto else (
                        "Detectado" if vector is not None else "Buscando...")
                    vis = dibujar_overlay(vis, coords, c, t, tipo=tipo)

            imgtk = _imgtk(vis, max_w, max_h)
            self.after(0, self._mostrar_frame, imgtk)
            time.sleep(0.033)

    def _mostrar_frame(self, imgtk):
        if not self.cam_running:
            return
        try:
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)
        except:
            pass

    def _loop_analisis(self):
        ultimo_id   = -1
        # Suavizado temporal: solo confirmar oclusion si se detecta
        # en N frames consecutivos. Evita falsos positivos por movimiento.
        _FRAMES_CONFIRM = 4   # frames seguidos necesarios para confirmar
        _oc_contador    = 0   # frames consecutivos con oclusion detectada
        _oc_razon_buf   = ""  # razon del ultimo frame con oclusion

        while self.cam_running:
            try:
                with self._frame_lock:
                    frame    = self._frame_actual
                    frame_id = id(frame) if frame is not None else -1

                if frame is None or frame_id == ultimo_id:
                    time.sleep(0.02)
                    continue

                ultimo_id = frame_id
                vector, coords, tipo = extraer_caracteristicas(
                    frame, HAAR_PATH,
                    modo=self._modo_deteccion,
                    tipo_esperado=self._tipo_esperado)

                ocluido_raw, razon_raw = False, ""
                if coords is not None:
                    try:
                        ocluido_raw, razon_raw = detectar_oclusion(frame, coords, tipo=tipo)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        ocluido_raw, razon_raw = False, ""

                # Suavizado temporal estable: +1 detecta / -1 no detecta
                # Requiere 4 frames consecutivos para confirmar (respuesta rapida)
                # Esto filtra falsos positivos por movimiento o ruido
                if ocluido_raw:
                    _oc_contador  = min(_oc_contador + 1, _FRAMES_CONFIRM)
                    _oc_razon_buf = razon_raw
                else:
                    _oc_contador  = max(_oc_contador - 1, 0)

                # Solo confirmar oclusion si se mantuvo N frames seguidos
                ocluido        = (_oc_contador >= _FRAMES_CONFIRM)
                razon_oclusion = _oc_razon_buf if ocluido else ""

                if ocluido:
                    vector = None

                with self._analisis_lock:
                    self._analisis["vector"]         = vector
                    self._analisis["coords"]         = coords
                    self._analisis["frame_id"]       = frame_id
                    self._analisis["tipo"]           = tipo
                    self._analisis["ocluido"]        = ocluido
                    self._analisis["razon_oclusion"] = razon_oclusion

            except Exception:
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _safe(self, fn):
        if self.cam_running:
            try: fn()
            except: pass

    # ══════════════════════════════════════════════════════════════════════════
    #  PANTALLA PRINCIPAL
    # ══════════════════════════════════════════════════════════════════════════
    def _build_main(self):
        self._clear()
        self.geometry(f"{W}x{H}+0+0")
        self.configure(bg=BG)

        cv = tk.Canvas(self, width=W, height=H, bg=BG, highlightthickness=0)
        cv.place(x=0, y=0)
        # Fondo suave institucional para que la pantalla no se vea plana
        cv.create_oval(-130, 100, 220, 450, fill="#E1F1EA", outline="")
        cv.create_oval(W-180, 420, W+160, 780, fill="#E5E0D4", outline="")
        cv.create_polygon(0, H-190, W, H-320, W, H, 0, H, fill="#EDE7DA", outline="")

        # ── Barra superior blanca (logo UdeC + título + cerrar) ───────────────
        TOP_H = 64
        cv.create_rectangle(0, 0, W, TOP_H, fill="#FFFFFF", outline="")
        # Separador inferior
        cv.create_rectangle(0, TOP_H - 1, W, TOP_H, fill=BORDER, outline="")

        self._logo_img = None
        try:
            import os as _os
            _base    = _os.path.dirname(_os.path.abspath(__file__))
            _path    = _os.path.join(_base, "img", "UdeC_2L_izq_Negro.png")
            _logo_pil = Image.open(_path).convert("RGBA")
            _logo_h   = 50
            _logo_w   = int(_logo_pil.width * _logo_h / _logo_pil.height)
            _logo_pil = _logo_pil.resize((_logo_w, _logo_h), Image.LANCZOS)
            _bg = Image.new("RGBA", (_logo_w, _logo_h), (255, 255, 255, 255))
            _bg.paste(_logo_pil, (0, 0), _logo_pil)
            self._logo_img = ImageTk.PhotoImage(_bg.convert("RGB"))
            cv.create_image(10, (TOP_H - _logo_h) // 2,
                            anchor="nw", image=self._logo_img)
        except Exception:
            pass

        # Título central (SICEUC)
        cv.create_text(W // 2, 32, text="SICEUC",
                       font=("Segoe UI", 13, "bold"), fill=NAVY_LN)

        # ── Botón SALIR (visible, con icono X y texto) ───────────────────────
        close_btn = tk.Button(
            self, text="✕  Salir",
            font=("Segoe UI", 9, "bold"),
            fg="#C1121F", bg="#FFF0F0",
            relief="flat", cursor="hand2",
            bd=0, highlightthickness=1,
            highlightbackground="#C1121F",
            command=self.on_close)
        close_btn.place(x=W - 80, y=14, width=68, height=28)
        close_btn.bind("<Enter>", lambda e: close_btn.config(bg="#C1121F", fg="#FFFFFF"))
        close_btn.bind("<Leave>", lambda e: close_btn.config(bg="#FFF0F0", fg="#C1121F"))

        # ── Arrastre de ventana (drag) ────────────────────────────────────────
        self._drag_x = 0; self._drag_y = 0
        def _drag_start(e):
            self._drag_x = e.x_root - self.winfo_x()
            self._drag_y = e.y_root - self.winfo_y()
        def _drag_move(e):
            self.geometry(f"+{e.x_root - self._drag_x}+{e.y_root - self._drag_y}")
        cv.tag_bind("drag_zone", "<ButtonPress-1>",   _drag_start)
        cv.tag_bind("drag_zone", "<B1-Motion>",       _drag_move)
        # Área invisible de arrastre sobre el header blanco (evita el botón cerrar)
        cv.create_rectangle(0, 0, W - 48, TOP_H, fill="", outline="",
                            tags="drag_zone")

        # ── Segunda barra: fondo azul marino + acento verde biselado ───────────
        SUB_H = 46
        SUB_Y = TOP_H
        cv.create_rectangle(0, SUB_Y, W, SUB_Y + SUB_H, fill=NAVY_LN, outline="")

        # Rectángulo verde con lado derecho en diagonal (bisel/paralelo)
        SKEW  = 18   # inclinación del lado derecho en px
        REC_W = 210  # ancho total del bloque verde
        cv.create_polygon(
            0,            SUB_Y,
            REC_W,        SUB_Y,
            REC_W - SKEW, SUB_Y + SUB_H,
            0,            SUB_Y + SUB_H,
            fill=TEAL_LN, outline="")

        # Texto del sistema sobre la barra
        cv.create_text(14, SUB_Y + 23, text="SISTEMA DE CONTROL DE ACCESO FACIAL",
                       font=("Segoe UI", 11, "bold"), fill="#FFFFFF", anchor="w")
        cv.create_text(W - 14, SUB_Y + 23,
                       text="Facultad de Ingeniería Electromecánica",
                       font=("Segoe UI", 9), fill="#AABBCC", anchor="e")

        # ── Zona central: título selección ────────────────────────────────────
        CONTENT_Y = SUB_Y + SUB_H + 28
        cv.create_text(W // 2, CONTENT_Y,
                       text="Selecciona una opción",
                       font=("Segoe UI", 17, "bold"), fill=TEXT)
        cv.create_text(W // 2, CONTENT_Y + 24,
                       text="Sistema de identificación biométrica — Universidad de Colima",
                       font=("Segoe UI", 8), fill=SUBTEXT)
        # Separador
        cv.create_line(20, CONTENT_Y + 40, W - 20, CONTENT_Y + 40,
                       fill=BORDER, width=1)

        # ── Tarjetas horizontales ─────────────────────────────────────────────
        BW, BH = W - 40, 148
        BX = 20
        BY1 = CONTENT_Y + 58
        BY2 = BY1 + BH + 22
        BY3 = BY2 + BH + 22

        self._horiz_card_btn(cv, BX, BY1, BW, BH,
                             "Registrar",
                             "Nuevo usuario",
                             "Captura biométrica guiada en 4 ángulos",
                             NAVY_LN, self._show_registro)
        self._horiz_card_btn(cv, BX, BY2, BW, BH,
                             "Acceso",
                             "Verificar identidad",
                             "Reconocimiento facial en tiempo real",
                             TEAL_LN, self._show_acceso)
        self._horiz_card_btn(cv, BX, BY3, BW, BH,
                             "Dashboard",
                             "Panel de administración",
                             "Gestión de usuarios, registros y estadísticas",
                             GOLD_LN, self._show_dashboard)

        # ── Barra inferior ────────────────────────────────────────────────────
        cv.create_rectangle(0, H - 32, W, H, fill=NAVY_LN, outline="")
        cv.create_text(W // 2, H - 16,
                       text=f"Universidad de Colima · v5.5 · {int(TIEMPO_ESCANEO)}s por ciclo · máx {MAX_MUESTRAS_PASO} muestras/paso",
                       font=("Segoe UI", 7), fill="#AABBCC")

    def _ir_a_dashboard(self):
        """Abre el dashboard si hay un usuario autenticado con rol permitido."""
        usuario = getattr(self, '_usuario_login', None)
        if usuario and usuario.get('rol') in ('admin', 'maestro'):
            self._abrir_dashboard(usuario)
        else:
            self._show_dashboard()

    def _card_btn(self, cv, x, y, w, h, titulo, subtitulo, desc, color, cmd):
        r = 18
        # Sombra suave
        _round_rect(cv, x+4, y+4, x+w+4, y+h+4,
                    r=r, fill="#080A10", outline="#080A10", width=0)
        # Cuerpo de la tarjeta
        _round_rect(cv, x, y, x+w, y+h,
                    r=r, fill=CARD, outline=BORDER, width=1)
        # Franja de color superior (acento)
        cv.create_rectangle(x+r, y, x+w-r, y+5, fill=color, outline="")
        cv.create_arc(x,     y,     x+2*r, y+2*r, start=90, extent=90,
                      fill=color, outline=color)
        cv.create_arc(x+w-2*r, y, x+w, y+2*r, start=0, extent=90,
                      fill=color, outline=color)
        cv.create_rectangle(x, y+r, x+w, y+5, fill=color, outline="")

        fr = tk.Frame(self, bg=CARD, width=w, height=h)
        fr.place(x=x, y=y)

        # Icono biométrico — círculo exterior + pupila
        ic = tk.Canvas(fr, width=52, height=52, bg=CARD, highlightthickness=0)
        ic.place(relx=.5, y=42, anchor="center")
        ic.create_oval(4,  4, 48, 48, outline=color, width=2)
        ic.create_oval(18, 18, 34, 34, fill=color, outline="")

        # Textos
        tk.Label(fr, text=titulo,
                 font=("Segoe UI", 13, "bold"),
                 fg=TEXT, bg=CARD
                 ).place(relx=.5, y=82, anchor="center")
        tk.Label(fr, text=subtitulo,
                 font=("Segoe UI", 8, "bold"),
                 fg=color, bg=CARD
                 ).place(relx=.5, y=102, anchor="center")
        tk.Label(fr, text=desc,
                 font=("Segoe UI", 8),
                 fg=SUBTEXT, bg=CARD, justify="center"
                 ).place(relx=.5, y=130, anchor="center")

        # Botón completamente redondeado (pill)
        btn_f = _rounded_btn(fr, text="ENTRAR  ▶",
                             cmd=cmd,
                             width=w - 48, height=36,
                             bg=color, fg=BG,
                             hover=_lighten_color(color),
                             font_size=9, bg_parent=CARD)
        btn_f.place(relx=.5, y=h - 30, anchor="center")

    def _horiz_card_btn(self, cv, x, y, w, h, titulo, subtitulo, desc, color, cmd):
        """
        Tarjeta horizontal pill — todo dibujado en un solo Canvas por tarjeta.
        Layout: [banda·icono redondeada izquierda | sep | textos | ENTRAR pill]
        """
        r       = 22      # radio bordes tarjeta
        band_w  = 120     # ancho banda colorida izquierda
        hover_c = _lighten_color(color)

        # Sin sombra externa para evitar barras visibles alrededor de las tarjetas

        # ── Canvas de la tarjeta (reemplaza Frame + Canvas separados) ─────────
        cc = tk.Canvas(self, width=w, height=h, bg=PANEL,
                       highlightthickness=0)
        cc.place(x=x, y=y)

        # Cuerpo tarjeta redondeado completo
        _round_rect(cc, 0, 0, w, h,
                    r=r, fill=PANEL, outline=color, width=3)

        # ── Banda colorida izquierda (solo esquinas izquierdas redondeadas) ───
        bw = band_w
        # Relleno de la banda: arcos izq + rectángulos
        cc.create_arc(0,      0,      2*r, 2*r,   start=90,  extent=90,
                      fill=color, outline=color)
        cc.create_arc(0,      h-2*r,  2*r, h,     start=180, extent=90,
                      fill=color, outline=color)
        cc.create_rectangle(r,  0,  bw, h,   fill=color, outline="")
        cc.create_rectangle(0,  r,  r,  h-r, fill=color, outline="")
        # Tapa el borde del CARD que quedó en la esquina superior izquierda
        cc.create_rectangle(0, 0, r, r, fill=PANEL, outline="")
        cc.create_rectangle(0, h-r, r, h, fill=PANEL, outline="")
        # Redibuja arcos del cuerpo sobre la esquina para limpiar
        cc.create_arc(0, 0,    2*r, 2*r,   start=90,  extent=90,
                      fill=color, outline=color)
        cc.create_arc(0, h-2*r, 2*r, h,    start=180, extent=90,
                      fill=color, outline=color)
        cc.create_rectangle(0, r,  r, h-r, fill=color, outline="")

        # ── Ícono intuitivo centrado en la banda ─────────────────────────────
        ic_cx, ic_cy = bw // 2, h // 2
        if "REGISTRAR" in titulo or "Nuevo" in subtitulo:
            # Ícono: clipboard / hoja de registro
            # Cuerpo del clipboard (rectángulo redondeado)
            cc.create_rectangle(ic_cx-22, ic_cy-30, ic_cx+22, ic_cy+32,
                                fill="#FFFFFF", outline="", )
            # Esquinas redondeadas del clipboard
            cc.create_arc(ic_cx-22, ic_cy-30, ic_cx-14, ic_cy-22,
                          start=90, extent=90, fill="#FFFFFF", outline="")
            cc.create_arc(ic_cx+14, ic_cy-30, ic_cx+22, ic_cy-22,
                          start=0, extent=90, fill="#FFFFFF", outline="")
            cc.create_arc(ic_cx-22, ic_cy+24, ic_cx-14, ic_cy+32,
                          start=180, extent=90, fill="#FFFFFF", outline="")
            cc.create_arc(ic_cx+14, ic_cy+24, ic_cx+22, ic_cy+32,
                          start=270, extent=90, fill="#FFFFFF", outline="")
            # Clip superior (lengüeta)
            cc.create_rectangle(ic_cx-10, ic_cy-34, ic_cx+10, ic_cy-24,
                                fill="#FFFFFF", outline="")
            cc.create_arc(ic_cx-10, ic_cy-38, ic_cx+10, ic_cy-28,
                          start=0, extent=180, fill="#FFFFFF", outline="")
            # Líneas de texto en el clipboard
            for i, lx2 in enumerate([14, 18, 10]):
                ly = ic_cy - 14 + i * 12
                cc.create_rectangle(ic_cx-14, ly, ic_cx+lx2, ly+4,
                                    fill=color, outline="")
            # Símbolo + abajo-derecha
            cc.create_oval(ic_cx+8, ic_cy+18, ic_cx+30, ic_cy+40,
                           fill=color, outline="#FFFFFF", width=2)
            cc.create_line(ic_cx+19, ic_cy+23, ic_cx+19, ic_cy+35,
                           fill="#FFFFFF", width=3)
            cc.create_line(ic_cx+13, ic_cy+29, ic_cx+25, ic_cy+29,
                           fill="#FFFFFF", width=3)
        elif "Dashboard" in titulo or "administración" in subtitulo or "estadísticas" in desc:
            # Ícono: barras de estadisticas (dashboard)
            # Fondo de "monitor"
            cc.create_rectangle(ic_cx-30, ic_cy-30, ic_cx+30, ic_cy+22,
                                fill="#FFFFFF", outline="")
            # Línea base del monitor
            cc.create_rectangle(ic_cx-30, ic_cy+22, ic_cx+30, ic_cy+26,
                                fill="#FFFFFF", outline="")
            # Pata del monitor
            cc.create_rectangle(ic_cx-4, ic_cy+26, ic_cx+4, ic_cy+34,
                                fill="#FFFFFF", outline="")
            cc.create_rectangle(ic_cx-12, ic_cy+34, ic_cx+12, ic_cy+38,
                                fill="#FFFFFF", outline="")
            # Barras de grafica de estadisticas (3 barras de distintas alturas)
            bar_w = 6
            bar_y_base = ic_cy + 14
            heights    = [16, 26, 20]  # altura de cada barra
            x_starts   = [ic_cx - 22, ic_cx - 6, ic_cx + 10]
            for bx, bh in zip(x_starts, heights):
                cc.create_rectangle(bx, bar_y_base - bh,
                                    bx + bar_w, bar_y_base,
                                    fill=color, outline="")
            # Linea de tendencia (flecha subiendo)
            cc.create_line(ic_cx - 22, ic_cy - 6,
                           ic_cx - 6,  ic_cy - 14,
                           ic_cx + 10, ic_cy - 10,
                           ic_cx + 22, ic_cy - 22,
                           fill=color, width=2, smooth=True)
            # Punta de flecha
            cc.create_polygon(ic_cx + 22, ic_cy - 22,
                              ic_cx + 16, ic_cy - 22,
                              ic_cx + 20, ic_cy - 16,
                              fill=color, outline="")
        else:
            # Ícono: cara con escáner / check de acceso
            # Cara oval
            cc.create_oval(ic_cx-26, ic_cy-34, ic_cx+26, ic_cy+18,
                           fill="#FFFFFF", outline="")
            # Ojos
            cc.create_oval(ic_cx-14, ic_cy-20, ic_cx-6,  ic_cy-12,
                           fill=color, outline="")
            cc.create_oval(ic_cx+6,  ic_cy-20, ic_cx+14, ic_cy-12,
                           fill=color, outline="")
            # Sonrisa
            cc.create_arc(ic_cx-12, ic_cy-8, ic_cx+12, ic_cy+10,
                          start=200, extent=140, style="arc",
                          outline=color, width=2)
            # Check badge abajo-derecha
            cc.create_oval(ic_cx+8, ic_cy+6, ic_cx+32, ic_cy+30,
                           fill="#1B2A4A", outline="#FFFFFF", width=2)
            cc.create_line(ic_cx+14, ic_cy+18, ic_cx+19, ic_cy+24,
                           fill="#FFFFFF", width=3)
            cc.create_line(ic_cx+19, ic_cy+24, ic_cx+27, ic_cy+13,
                           fill="#FFFFFF", width=3)

        # ── Línea separadora vertical ─────────────────────────────────────────
        cc.create_line(bw+10, 14, bw+10, h-14, fill=BORDER, width=1)

        # ── Textos ────────────────────────────────────────────────────────────
        tx = bw + 26
        cc.create_text(tx, 28,  text=titulo,    anchor="w",
                       font=("Segoe UI", 16, "bold"), fill=TEXT)
        cc.create_text(tx, 56,  text=subtitulo, anchor="w",
                       font=("Segoe UI", 9, "bold"),  fill=color)
        cc.create_text(tx, 76,  text=desc,      anchor="w",
                       font=("Segoe UI", 8),          fill=SUBTEXT)

        # ── Botón pill ENTRAR (Canvas embebido via create_window) ─────────────
        btn_w2, btn_h2 = 150, 46
        btn_f = _rounded_btn(cc, "ENTRAR  ▶", cmd,
                             width=btn_w2, height=btn_h2,
                             bg=color, fg="#FFFFFF",
                             hover=hover_c,
                             font_size=9, bg_parent=PANEL)
        cc.create_window(w - btn_w2 - 18, (h - btn_h2) // 2,
                         anchor="nw", window=btn_f)

        # ── Card completa clickable ───────────────────────────────────────────
        cc.bind("<Button-1>", lambda e: cmd())

    @staticmethod
    def _lighten_static(hx):
        h = hx.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{min(255,r+40):02x}{min(255,g+40):02x}{min(255,b+40):02x}"

    def _crear_fondo_camara(self, x, y, w, h, titulo="Cámara activa", subtitulo="Coloca tu rostro dentro del recuadro"):
        """Fondo visual para las pantallas con cámara. Da contexto si la cámara tarda en pintar."""
        cv = tk.Canvas(self, width=w, height=h, bg="#0B1120", highlightthickness=0)
        cv.place(x=x, y=y)
        # Fondo tipo panel con líneas suaves, sin depender de imágenes externas
        cv.create_rectangle(0, 0, w, h, fill="#07101F", outline="")
        cv.create_oval(-110, -90, 230, 215, fill="#102B3E", outline="")
        cv.create_oval(w-210, h-190, w+130, h+145, fill="#12351F", outline="")
        cv.create_oval(w//2-260, h//2-210, w//2+260, h//2+230, fill="#0B1A2D", outline="")
        cv.create_rectangle(0, 0, w, 5, fill=TEAL_LN, outline="")
        # Marco de cámara
        m = 8
        _round_rect(cv, m, m, w-m, h-m, r=20, fill="#0E1A2B", outline="#1E3A5F", width=2)
        # Guía central punteada
        cx, cy = w // 2, h // 2
        size = min(w, h) // 3
        cv.create_line(cx-size, cy-size, cx-size+45, cy-size, fill="#355F88", width=2)
        cv.create_line(cx-size, cy-size, cx-size, cy-size+45, fill="#355F88", width=2)
        cv.create_line(cx+size, cy-size, cx+size-45, cy-size, fill="#355F88", width=2)
        cv.create_line(cx+size, cy-size, cx+size, cy-size+45, fill="#355F88", width=2)
        cv.create_line(cx-size, cy+size, cx-size+45, cy+size, fill="#355F88", width=2)
        cv.create_line(cx-size, cy+size, cx-size, cy+size-45, fill="#355F88", width=2)
        cv.create_line(cx+size, cy+size, cx+size-45, cy+size, fill="#355F88", width=2)
        cv.create_line(cx+size, cy+size, cx+size, cy+size-45, fill="#355F88", width=2)
        cv.create_text(cx, cy-10, text=titulo, font=("Segoe UI", 12, "bold"), fill="#DCEBFF")
        cv.create_text(cx, cy+16, text=subtitulo, font=("Segoe UI", 8), fill="#8FA7C8")
        return cv

    def _crear_teclado_virtual(self, parent, x, y, w, h, FONT="Segoe UI"):
        """Teclado táctil simple para pantallas sin teclado físico."""
        self._vk_shift = False
        if not hasattr(self, "_vk_target"):
            self._vk_target = None

        kb = tk.Frame(parent, bg="#EAF0F7", width=w, height=h)
        kb.place(x=x, y=y)
        kb.pack_propagate(False)

        header = tk.Frame(kb, bg="#EAF0F7", height=28)
        header.pack(fill="x")
        tk.Label(header, text="⌨  Teclado táctil", font=(FONT, 8, "bold"),
                 fg=NAVY_LN, bg="#EAF0F7", anchor="w").pack(side="left", padx=10)
        tk.Label(header, text="Toca un campo y escribe aquí", font=(FONT, 7),
                 fg=SUBTEXT, bg="#EAF0F7", anchor="e").pack(side="right", padx=10)

        rows_holder = tk.Frame(kb, bg="#EAF0F7")
        rows_holder.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        def _target():
            ent = getattr(self, "_vk_target", None)
            try:
                if ent is not None and ent.winfo_exists():
                    ent.focus_set()
                    return ent
            except Exception:
                pass
            return None

        def _insert(txt):
            ent = _target()
            if ent is None:
                return
            try:
                pos = ent.index(tk.INSERT)
                ent.insert(pos, txt)
            except Exception:
                pass

        def _backspace():
            ent = _target()
            if ent is None:
                return
            try:
                pos = ent.index(tk.INSERT)
                if pos > 0:
                    ent.delete(pos-1, pos)
            except Exception:
                pass

        def _clear():
            ent = _target()
            if ent is None:
                return
            try:
                ent.delete(0, tk.END)
            except Exception:
                pass

        def _toggle_shift():
            self._vk_shift = not getattr(self, "_vk_shift", False)
            _draw_keys()

        def _press(key):
            if key == "⌫":
                _backspace()
            elif key == "Limpiar":
                _clear()
            elif key == "Espacio":
                _insert(" ")
            elif key == "⇧":
                _toggle_shift()
            elif key == ".com":
                _insert(".com")
            else:
                if len(key) == 1 and key.isalpha():
                    _insert(key.upper() if getattr(self, "_vk_shift", False) else key.lower())
                else:
                    _insert(key)

        def _mk_button(row, text, weight=1, bg="#FFFFFF", fg=TEXT):
            b = tk.Button(row, text=text, font=(FONT, 9, "bold"), fg=fg, bg=bg,
                          activebackground="#DCEFE7", activeforeground=TEXT,
                          relief="flat", bd=0, highlightthickness=1,
                          highlightbackground="#CFD8E6", cursor="hand2",
                          command=lambda k=text: _press(k))
            b.pack(side="left", fill="both", expand=True, padx=3, pady=3)
            return b

        def _draw_keys():
            for child in rows_holder.winfo_children():
                child.destroy()
            layouts = [
                list("1234567890"),
                list("qwertyuiop"),
                list("asdfghjklñ"),
                ["⇧"] + list("zxcvbnm") + ["⌫"],
                ["@", "_", "-", ".", "/", "Espacio", ".com", "Limpiar"],
            ]
            for layout in layouts:
                row = tk.Frame(rows_holder, bg="#EAF0F7")
                row.pack(fill="x", expand=True)
                for key in layout:
                    show = key
                    if len(key) == 1 and key.isalpha():
                        show = key.upper() if getattr(self, "_vk_shift", False) else key.lower()
                    if key == "⇧":
                        _mk_button(row, show, bg=ACCENT if getattr(self, "_vk_shift", False) else "#FFFFFF",
                                   fg="#FFFFFF" if getattr(self, "_vk_shift", False) else TEXT)
                    elif key in ("⌫", "Limpiar"):
                        _mk_button(row, show, bg="#FFECEC", fg=DANGER)
                    elif key in ("Espacio", ".com"):
                        _mk_button(row, show, bg="#EAF7F0", fg=ACCENT)
                    else:
                        _mk_button(row, show)
        _draw_keys()
        return kb

    # ══════════════════════════════════════════════════════════════════════════
    #  PANTALLA REGISTRO
    # ══════════════════════════════════════════════════════════════════════════
    def _show_registro(self):
        # Asegurar que cualquier cámara previa esté liberada antes de reiniciar
        self._stop_cam()
        # ── CRITICO: resetear estado de acceso al entrar al registro ──────────
        # Si veniamos de la pantalla de Acceso, _modo_acceso seguia True.
        # Eso causaba dos bugs:
        #   1. El _loop_camara dibujaba el bbox con la rama de acceso
        #      (sin fallback de texto, sin label de pose) → el cuadrito
        #      y el texto "FRENTE/IZQUIERDA/..." desaparecian.
        #   2. Si Acceso habia programado _lanzar_verificacion via after(),
        #      el callback se disparaba aunque ya estuvieras en Registro,
        #      llamando reconocer_persona y "reconociendo" a la persona
        #      que se estaba registrando.
        self._modo_acceso = False
        self.verificando  = False
        self._clear()
        SCREEN_H = self.winfo_screenheight()
        WIN_H    = min(H, SCREEN_H)
        self.geometry(f"{W}x{WIN_H}+0+0")
        self._set_overlay(None, "")
        self._registro_cancelado = False

        FONT   = "Segoe UI"
        HDR_H  = 50
        PILL_H = 34
        PILL_Y = HDR_H + 10
        CAM_Y  = PILL_Y + PILL_H + 8
        BOT_H  = 120
        CAM_H  = WIN_H - CAM_Y - BOT_H

        # ── Header oscuro ─────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=NAVY_LN, width=W, height=HDR_H)
        hdr.place(x=0, y=0)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="REGISTRO", font=(FONT, 13, "bold"),
                 fg="#FFFFFF", bg=NAVY_LN).place(x=18, y=14)
        menu_hdr = tk.Button(hdr, text="☰  Menú",
                              font=(FONT, 8, "bold"), fg="#FFFFFF", bg=TEAL_LN,
                              relief="flat", cursor="hand2", bd=0,
                              highlightthickness=0,
                              command=lambda: [self._stop_cam(), self._build_main()])
        menu_hdr.place(x=W - 90, y=12, width=78, height=26)
        menu_hdr.bind("<Enter>", lambda e: menu_hdr.config(bg="#1A6FD4"))
        menu_hdr.bind("<Leave>", lambda e: menu_hdr.config(bg=TEAL_LN))
        menu_hdr.lift()

        # ── Franja beige entre header y cámara ────────────────────────────────
        tk.Frame(self, bg=BG, width=W, height=PILL_H + 18).place(x=0, y=HDR_H)

        # ── Pill de instrucción ───────────────────────────────────────────────
        PILL_W = 260
        self.pill_cv_reg = tk.Canvas(self, width=PILL_W, height=PILL_H,
                                     bg=BG, highlightthickness=0)
        self.pill_cv_reg.place(x=W // 2 - PILL_W // 2, y=PILL_Y)
        self.posicion_var = tk.StringVar(value="Posiciónate frente a la cámara")

        def _draw_pill_reg(color):
            r = PILL_H // 2
            self.pill_cv_reg.delete("all")
            self.pill_cv_reg.create_arc(0, 0, 2*r, PILL_H,
                                        start=90, extent=180, fill=color, outline=color)
            self.pill_cv_reg.create_arc(PILL_W - 2*r, 0, PILL_W, PILL_H,
                                        start=270, extent=180, fill=color, outline=color)
            self.pill_cv_reg.create_rectangle(r, 0, PILL_W - r, PILL_H,
                                              fill=color, outline="")
            try:
                self._pill_reg_lbl.config(bg=color)
            except Exception:
                pass

        _draw_pill_reg(ACCENT2)
        self._draw_pill_reg = _draw_pill_reg
        self._pill_reg_lbl = tk.Label(
            self.pill_cv_reg, textvariable=self.posicion_var,
            font=(FONT, 9, "bold"), fg="#FFFFFF",
            bg=ACCENT2, wraplength=PILL_W - 16)
        self._pill_reg_lbl.place(relx=.5, rely=.5, anchor="center")

        # ── Zona cámara con fondo visual institucional ─────────────────────────
        self._crear_fondo_camara(0, CAM_Y, W, CAM_H,
                                 "Escaneo facial",
                                 "Llena tus datos y mantén el rostro visible")
        self.cam_label = tk.Label(self, bg="#0B1120", bd=0, highlightthickness=0)
        self.cam_label.place(x=8, y=CAM_Y + 8, width=W - 16, height=CAM_H - 16)

        # ── Botón INICIAR ESCANEO (sobre la cámara, parte baja) ───────────────
        self._form_listo = False
        BTN_W, BTN_H_b = W - 32, 42
        BTN_Y_b = CAM_Y + CAM_H - BTN_H_b - 14

        self.scan_btn_cv = tk.Canvas(self, width=BTN_W, height=BTN_H_b,
                                     bg="#1A1A1A", highlightthickness=0)
        self.scan_btn_cv.place(x=16, y=BTN_Y_b)

        def _draw_scan_btn(listo):
            self.scan_btn_cv.delete("all")
            color = ACCENT    if listo else "#1E2E1E"
            tc    = "#FFFFFF" if listo else "#3A5A3A"
            r = BTN_H_b // 2
            self.scan_btn_cv.create_arc(0, 0, r*2, BTN_H_b,
                                        start=90, extent=180, fill=color, outline=color)
            self.scan_btn_cv.create_arc(BTN_W - r*2, 0, BTN_W, BTN_H_b,
                                        start=270, extent=180, fill=color, outline=color)
            self.scan_btn_cv.create_rectangle(r, 0, BTN_W - r, BTN_H_b,
                                              fill=color, outline="")
            icono = "⬤  INICIAR ESCANEO" if listo else "🔒  Completa el formulario para continuar"
            self.scan_btn_cv.create_text(BTN_W // 2, BTN_H_b // 2,
                                         text=icono, font=(FONT, 10, "bold"), fill=tc)

        _draw_scan_btn(False)
        self._draw_scan_btn = _draw_scan_btn

        def _on_scan_click(e):
            if self._form_listo:
                self._lanzar_desde_btn()

        self.scan_btn_cv.bind("<Button-1>", _on_scan_click)
        self.scan_btn_cv.bind("<Enter>",
            lambda e: _draw_scan_btn(True) if self._form_listo else None)
        self.scan_btn_cv.bind("<Leave>",
            lambda e: _draw_scan_btn(self._form_listo))

        # ── Panel inferior oscuro ─────────────────────────────────────────────
        scan_bot = tk.Frame(self, bg=NAVY_LN, width=W, height=BOT_H)
        scan_bot.place(x=0, y=WIN_H - BOT_H)
        scan_bot.pack_propagate(False)
        tk.Frame(scan_bot, bg=TEAL_LN, width=W, height=3).place(x=0, y=0)

        self.reg_nombre_var  = tk.StringVar(value="")
        self.reg_detalle_var = tk.StringVar(value="")
        tk.Label(scan_bot, textvariable=self.reg_nombre_var,
                 font=(FONT, 10, "bold"), fg="#FFFFFF", bg=NAVY_LN,
                 anchor="w").place(x=14, y=8, width=W - 28)
        tk.Label(scan_bot, textvariable=self.reg_detalle_var,
                 font=(FONT, 8), fg="#AABBCC", bg=NAVY_LN,
                 anchor="w").place(x=14, y=28, width=W - 120)

        # Pasos (círculos 1-4)
        CIRCLE_R    = 11
        step_labels = ["F", "I", "D", "F"]
        step_names  = ["Frente", "Izq.", "Der.", "Frente"]
        STEP_GAP    = (W - 28 - 4 * CIRCLE_R * 2) // 5
        self._paso_frames = []

        paso_cv = tk.Canvas(scan_bot, width=W - 28, height=48,
                            bg=NAVY_LN, highlightthickness=0)
        paso_cv.place(x=14, y=52)

        paso_cv.create_line(CIRCLE_R * 2 + STEP_GAP, CIRCLE_R,
                            W - 28 - CIRCLE_R * 2 - STEP_GAP, CIRCLE_R,
                            fill="#2A4060", width=2)

        self._paso_canvas  = paso_cv
        self._paso_cx      = []
        self._paso_cy      = []
        self._paso_circles = []
        self._paso_nums    = []
        self._paso_prog    = []

        for i in range(4):
            cx = STEP_GAP + CIRCLE_R + i * (CIRCLE_R * 2 + STEP_GAP)
            cy = CIRCLE_R
            cid = paso_cv.create_oval(cx - CIRCLE_R, cy - CIRCLE_R,
                                      cx + CIRCLE_R, cy + CIRCLE_R,
                                      fill="#2A4060", outline="#2A4060", width=2)
            nid = paso_cv.create_text(cx, cy, text=step_labels[i],
                                      font=(FONT, 7, "bold"), fill="#AABBCC")
            paso_cv.create_text(cx, cy + CIRCLE_R + 9,
                                text=step_names[i], font=(FONT, 6), fill="#5577AA")
            cnt = paso_cv.create_text(cx, cy + CIRCLE_R + 19,
                                      text="", font=(FONT, 6), fill="#5577AA")
            self._paso_cx.append(cx)
            self._paso_cy.append(cy)
            self._paso_circles.append(cid)
            self._paso_nums.append(nid)
            self._paso_prog.append(cnt)
            self._paso_frames.append((None, None, None, None))

        # Barra de progreso pegada al fondo del panel
        self.prog_frame = tk.Frame(scan_bot, bg="#2A4060", width=W - 28, height=4)
        self.prog_frame.place(x=14, y=BOT_H - 6)
        self.prog_bar = tk.Frame(self.prog_frame, bg=ACCENT, width=0, height=4)
        self.prog_bar.place(x=0, y=0)

        # Botón CANCELAR (dentro del panel, aparece durante escaneo)
        self._cancel_btn = tk.Button(
            scan_bot, text="✕  Cancelar registro",
            font=(FONT, 8, "bold"), fg=DANGER, bg=NAVY_LN,
            relief="flat", bd=0, highlightthickness=0, cursor="hand2",
            command=self._cancelar_registro)
        self._cancel_btn.place(x=W - 150, y=24, width=136, height=20)
        self._cancel_btn.place_forget()  # oculto hasta que inicie el escaneo

        # Variables de compatibilidad
        self.paso_desc_var  = tk.StringVar(value="")
        self._paso_desc_lbl = tk.Label(scan_bot, textvariable=self.paso_desc_var,
                                       font=(FONT, 7), fg=ACCENT2, bg=NAVY_LN)
        self.status_var.set("Listo")
        self.progreso_var = tk.StringVar(value="")
        self.prog_label   = tk.Label(scan_bot, textvariable=self.progreso_var,
                                     font=(FONT, 7, "bold"), fg=SUCCESS,
                                     bg=NAVY_LN, anchor="w")
        self.prog_label.place(x=14, y=BOT_H - 20, width=W - 28)
        self.timer_var    = tk.StringVar(value="")
        self.paso_txt_var = tk.StringVar(value="")

        # Barras de muestras (compatibilidad)
        self._barra_pasos = []
        for i in range(4):
            bar = tk.Frame(scan_bot, bg=BORDER, width=0, height=3)
            lbl = tk.Label(scan_bot, text="0", font=("Segoe UI", 7), fg=SUBTEXT, bg=NAVY_LN)
            self._barra_pasos.append((bar, lbl, 60))

        # ── Botón grande "Datos del usuario" ─────────────────────────────────
        # En pantalla táctil debe verse como una acción clara, no como icono pequeño.
        FAB_W, FAB_H = 226, 50
        FAB_X = W - FAB_W - 16
        FAB_Y = BTN_Y_b - FAB_H - 10

        fab_cv = tk.Canvas(self, width=FAB_W, height=FAB_H,
                           bg="#0B1120", highlightthickness=0)
        fab_cv.place(x=FAB_X, y=FAB_Y)
        self._fab_cv = fab_cv

        def _draw_fab(color=NAVY_LN):
            fab_cv.delete("all")
            r = FAB_H // 2
            # sombra / resplandor
            fab_cv.create_arc(3, 5, 3+2*r, FAB_H+5, start=90, extent=180,
                              fill="#000000", outline="#000000")
            fab_cv.create_arc(FAB_W-2*r+3, 5, FAB_W+3, FAB_H+5, start=270, extent=180,
                              fill="#000000", outline="#000000")
            fab_cv.create_rectangle(r+3, 5, FAB_W-r+3, FAB_H+5,
                                    fill="#000000", outline="")
            # cuerpo
            fab_cv.create_arc(0, 0, 2*r, FAB_H, start=90, extent=180,
                              fill=color, outline=color)
            fab_cv.create_arc(FAB_W-2*r, 0, FAB_W, FAB_H, start=270, extent=180,
                              fill=color, outline=color)
            fab_cv.create_rectangle(r, 0, FAB_W-r, FAB_H, fill=color, outline="")
            # icono clipboard
            cx, cy = 26, FAB_H // 2
            fab_cv.create_rectangle(cx-9, cy-13, cx+9, cy+13,
                                    fill="#FFFFFF", outline="")
            fab_cv.create_rectangle(cx-5, cy-17, cx+5, cy-11,
                                    fill="#FFFFFF", outline="")
            for dy in (-5, 1, 7):
                fab_cv.create_line(cx-6, cy+dy, cx+6, cy+dy,
                                   fill=color, width=2)
            fab_cv.create_text(48, 16, text="DATOS DEL USUARIO",
                               font=(FONT, 9, "bold"), fill="#FFFFFF", anchor="w")
            fab_cv.create_text(48, 33, text="Toca aquí para abrir formulario",
                               font=(FONT, 7), fill="#D8E8FF", anchor="w")

        _draw_fab(NAVY_LN)
        self._draw_fab = _draw_fab

        def _open_form():
            _draw_fab(TEAL_LN)
            self._show_form_sheet(WIN_H, FONT)

        fab_cv.bind("<Button-1>", lambda e: _open_form())
        fab_cv.bind("<Enter>",    lambda e: _draw_fab(TEAL_LN))
        fab_cv.bind("<Leave>",    lambda e: _draw_fab(NAVY_LN))

        # ── Iniciar cámara ────────────────────────────────────────────────────
        self._start_cam()
        self.cam_running = True
        threading.Thread(target=self._loop_camara,
                         kwargs={"max_w": W - 16, "max_h": CAM_H - 16},
                         daemon=True).start()
        threading.Thread(target=self._loop_analisis, daemon=True).start()

        # Auto-abrir formulario para que el usuario empiece llenandolo.
        # El FAB sigue disponible para editarlo despues si es necesario.
        self.after(450, lambda: self._show_form_sheet(WIN_H, FONT))

    def _cancelar_registro(self):
        """Detiene el escaneo en curso y regresa a la pantalla de registro."""
        self._registro_cancelado = True
        self.cam_running = False
        self.verificando = False
        self.after(200, self._show_registro)


    def _actualizar_campos_rol(self):
        """
        Muestra correo+contrasena para admin/maestro,
        o grado+grupo para estudiante.
        """
        for w in (self._lbl_correo, self._ent_correo,
                  self._lbl_pwd,    self._ent_pwd,
                  self._lbl_grado,  self._om_grado,
                  self._lbl_grupo,  self._om_grupo):
            try: w.place_forget()
            except: pass

        rol = self.rol_var.get()
        FW2 = (W - 36) // 2 - 4
        if rol in ("admin", "maestro"):
            self._lbl_correo.place(x=0,        y=2)
            self._ent_correo.place(x=0,        y=28, width=FW2, height=32)
            self._lbl_pwd.place(  x=FW2 + 8,  y=2)
            self._ent_pwd.place(  x=FW2 + 8,  y=28, width=FW2, height=32)
        else:  # estudiante
            self._lbl_grado.place(x=0,   y=2)
            self._om_grado.place( x=0,   y=28, width=100, height=32)
            self._lbl_grupo.place(x=118, y=2)
            self._om_grupo.place( x=118, y=28, width=100, height=32)

    def _field(self, parent, label, var, y, show=None):
        tk.Label(parent, text=label, font=self.f_label,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=y)
        kw = dict(font=self.f_sub, fg=TEXT, bg=CARD,
                  insertbackground=ACCENT, relief="flat",
                  highlightthickness=1, highlightcolor=ACCENT,
                  highlightbackground=BORDER)
        if show:
            kw["show"] = show
        tk.Entry(parent, textvariable=var, **kw
                 ).place(x=18, y=y+16, width=284, height=24)  # Más altura (24 en lugar de 22)

    def _activar_paso_ui(self, paso_idx, progreso=0.0):
        if not hasattr(self, "_paso_canvas") or not self.cam_running:
            return
        try:
            cv = self._paso_canvas
            for i in range(len(self._paso_circles)):
                cid = self._paso_circles[i]
                nid = self._paso_nums[i]
                cnt = self._paso_prog[i]
                cx  = self._paso_cx[i]
                cy  = self._paso_cy[i]
                r   = 14
                if i < paso_idx:
                    cv.itemconfig(cid, fill=SUCCESS, outline=SUCCESS)
                    cv.itemconfig(nid, fill="#FFFFFF")
                elif i == paso_idx:
                    cv.itemconfig(cid, fill=ACCENT, outline=ACCENT)
                    cv.itemconfig(nid, fill="#FFFFFF")
                else:
                    cv.itemconfig(cid, fill="#2A4060", outline="#2A4060")
                    cv.itemconfig(nid, fill="#AABBCC")
        except:
            pass

    def _resetear_pasos_ui(self):
        if not hasattr(self, "_paso_canvas"):
            return
        try:
            cv = self._paso_canvas
            for i in range(len(self._paso_circles)):
                cv.itemconfig(self._paso_circles[i], fill="#2A4060", outline="#2A4060")
                cv.itemconfig(self._paso_nums[i], fill="#AABBCC")
                cv.itemconfig(self._paso_prog[i], text="", fill="#5577AA")
        except:
            pass

    def _update_barra_paso(self, paso_idx, n_muestras):
        # Actualizar contador en el canvas de círculos
        if hasattr(self, "_paso_prog") and paso_idx < len(self._paso_prog):
            try:
                pct = min(1.0, n_muestras / MAX_MUESTRAS_PASO)
                col = SUCCESS if pct >= 1.0 else ACCENT if pct > 0.4 else WARNING
                self._paso_canvas.itemconfig(
                    self._paso_prog[paso_idx],
                    text=str(n_muestras), fill=col)
            except:
                pass
        # Barra de muestras (ahora invisible pero se mantiene la lógica)
        if not hasattr(self, "_barra_pasos") or \
                paso_idx >= len(self._barra_pasos):
            return
        try:
            bar, lbl, barra_w = self._barra_pasos[paso_idx]
            pct = min(1.0, n_muestras / MAX_MUESTRAS_PASO)
            w   = int(pct * barra_w)
            col = SUCCESS if pct >= 1.0 else ACCENT if pct > 0.4 else WARNING
            bar.config(width=w, bg=col)
            lbl.config(text=str(n_muestras), fg=col)
        except:
            pass

    def _cancelar_por_duplicado(self, duplicado: dict):
        # Overlay rojo en cámara con nombre del duplicado
        nombre_dup = duplicado.get("nombre", "Persona desconocida")
        self._set_overlay((255, 59, 92), f"YA REGISTRADO\n{nombre_dup}")
        try:
            self.progreso_var.set(f"Rostro ya registrado como:\n{nombre_dup}")
            self.prog_label.config(fg=DANGER, font=("Segoe UI", 10, "bold"))
            self.status_var.set("Registro cancelado.")
            self.timer_var.set("")
            self.paso_txt_var.set("")
        except:
            pass

        self.after(100, self._resetear_pasos_ui)

        # Restaurar estado normal después de 5 segundos
        def _restaurar():
            try:
                self.prog_label.config(font=("Segoe UI", 7, "bold"))
                self.progreso_var.set("")
                self.status_var.set("Listo")
                self._set_overlay(None, "")
                # Mostrar de nuevo el botón de escaneo y FAB
                try:
                    self.scan_btn_cv.place(x=16, y=self.scan_btn_cv.winfo_y())
                    self._fab_cv.place(x=16, y=self._fab_cv.winfo_y())
                    self._cancel_btn.place_forget()
                except Exception:
                    pass
            except:
                pass

        self.after(5000, _restaurar)


    def _show_form_sheet(self, WIN_H, FONT):
        """Bottom sheet con el formulario de registro."""
        # En pantalla vertical táctil damos más alto al sheet para que quepa
        # el formulario + teclado virtual sin tapar el botón de guardado.
        SHEET_H = min(WIN_H - 10, int(WIN_H * 0.95))

        # Overlay semitransparente
        overlay = tk.Frame(self, bg="#000000")
        overlay.place(x=0, y=0, width=W, height=WIN_H)
        overlay_cv = tk.Canvas(overlay, width=W, height=WIN_H,
                               bg="#000000", highlightthickness=0)
        overlay_cv.place(x=0, y=0)
        overlay_cv.create_rectangle(0, 0, W, WIN_H, fill="#000000", stipple="gray50")

        # Sheet frame
        sheet = tk.Frame(self, bg="#F8FAFC", width=W, height=SHEET_H)
        sheet.place(x=0, y=WIN_H)
        sheet.pack_propagate(False)

        # Título + botón cerrar
        tk.Label(sheet, text="📋  Datos del usuario",
                 font=(FONT, 13, "bold"), fg=TEXT, bg="#F8FAFC",
                 anchor="w").place(x=22, y=18)

        def _close_sheet():
            def _slide_down(y):
                if y <= WIN_H:
                    sheet.place(y=y)
                    self.after(12, lambda: _slide_down(y + 30))
                else:
                    sheet.destroy()
                    overlay.destroy()
                    try:
                        self._draw_fab(NAVY_LN)
                    except Exception:
                        pass
            _slide_down(sheet.winfo_y())

        close_btn = tk.Button(sheet, text="✕",
                              font=(FONT, 11, "bold"), fg=SUBTEXT, bg="#F8FAFC",
                              relief="flat", bd=0, highlightthickness=0,
                              cursor="hand2", command=_close_sheet)
        close_btn.place(x=W - 46, y=16, width=34, height=34)

        PAD  = 18
        COL2 = W // 2 + 4
        FW   = (W - 48) // 2 - 4
        touch_entries = []

        def campo(label_txt, var, x, y, show=None, w=FW):
            """Campo táctil redondeado con foco visual + teclado virtual."""
            tk.Label(sheet, text=label_txt, font=(FONT, 9, "bold"),
                     fg="#344054", bg="#F8FAFC").place(x=x, y=y)

            holder = tk.Frame(sheet, bg="#F8FAFC", width=w, height=40)
            holder.place(x=x, y=y + 24)
            holder.pack_propagate(False)
            bgcv = tk.Canvas(holder, width=w, height=40, bg="#F8FAFC", highlightthickness=0)
            bgcv.place(x=0, y=0)

            def _draw(border=BORDER, fill=CARD):
                bgcv.delete("all")
                _round_rect(bgcv, 1, 1, w-1, 39, r=16, fill=fill, outline=border, width=1)

            _draw()
            e = tk.Entry(holder, textvariable=var, font=(FONT, 11), fg=TEXT, bg=CARD,
                         insertbackground=ACCENT, relief="flat", bd=0,
                         highlightthickness=0)
            if show:
                e.config(show=show)
            e.place(x=14, y=8, width=w-28, height=24)

            def _set_target(_ev=None):
                self._vk_target = e
                try: e.focus_set()
                except Exception: pass
                _draw(ACCENT, "#F5FFF9")

            def _blur(_ev=None):
                _draw(BORDER, CARD)

            e.bind("<FocusIn>", _set_target)
            e.bind("<Button-1>", _set_target)
            e.bind("<FocusOut>", _blur)
            holder.bind("<Button-1>", _set_target)
            bgcv.bind("<Button-1>", _set_target)
            touch_entries.append(e)
            return e

        # Sección Identidad
        tk.Label(sheet, text="Identidad", font=(FONT, 9, "bold"),
                 fg=NAVY_LN, bg="#F8FAFC").place(x=PAD, y=56)

        campo("Nombre(s)",   self.nombre_var, PAD,  82, w=W - 36)
        campo("Ap. paterno", self.ap_pat_var, PAD,  150)
        campo("Ap. materno", self.ap_mat_var, COL2, 150)
        campo("Número de cuenta (8 dígitos)", self.cuenta_var, PAD, 218, w=W - 36)

        # Sección Rol
        tk.Label(sheet, text="Rol", font=(FONT, 9, "bold"),
                 fg=NAVY_LN, bg="#F8FAFC").place(x=PAD, y=296)

        rol_frame = tk.Frame(sheet, bg="#E9EEF6", width=W - 36, height=42)
        rol_frame.place(x=PAD, y=316)
        rol_frame.pack_propagate(False)

        # Estado en tiempo real
        estado_lbl = tk.Label(sheet, text="Estado: Formulario incompleto",
                              font=(FONT, 9, "bold"), fg=SUBTEXT, bg="#F8FAFC", anchor="w")
        estado_lbl.place(x=PAD, y=446, width=W - 2*PAD)

        # ── Botón GUARDAR (esquina inferior derecha) ─────────────────────────
        # Solo se habilita cuando el formulario es valido.
        # Al presionarlo cierra el sheet Y asegura que _form_listo este sincronizado.
        BTN_G_W, BTN_G_H = 178, 44
        guardar_cv = tk.Canvas(sheet, width=BTN_G_W, height=BTN_G_H,
                               bg="#F8FAFC", highlightthickness=0)
        guardar_cv.place(x=W - BTN_G_W - PAD,
                         y=SHEET_H - BTN_G_H - 14)

        def _draw_guardar(activo):
            guardar_cv.delete("all")
            r = BTN_G_H // 2
            col = SUCCESS if activo else "#2A4060"
            tc  = "#FFFFFF" if activo else "#5577AA"
            guardar_cv.create_arc(0, 0, 2*r, BTN_G_H,
                                  start=90, extent=180, fill=col, outline=col)
            guardar_cv.create_arc(BTN_G_W - 2*r, 0, BTN_G_W, BTN_G_H,
                                  start=270, extent=180, fill=col, outline=col)
            guardar_cv.create_rectangle(r, 0, BTN_G_W - r, BTN_G_H,
                                        fill=col, outline="")
            guardar_cv.create_text(BTN_G_W // 2, BTN_G_H // 2,
                                   text="✓  GUARDAR DATOS" if activo else "GUARDAR DATOS",
                                   font=(FONT, 10, "bold"), fill=tc)

        _draw_guardar(False)
        self._draw_guardar = _draw_guardar

        def _validar_y_actualizar(*args):
            """Valida los campos en tiempo real y desbloquea el botón de escaneo."""
            # Si el sheet fue cerrado, el label ya no existe — ignorar silenciosamente
            try:
                if not estado_lbl.winfo_exists():
                    return
            except Exception:
                return
            nombre = self.nombre_var.get().strip()
            ap_pat = self.ap_pat_var.get().strip()
            cuenta = self.cuenta_var.get().strip()
            correo = self.correo_var.get().strip()
            pwd    = self.pwd_var.get()
            rol    = self.rol_var.get()
            grado  = self.grado_var.get()
            grupo  = self.grupo_var.get()

            def _set_invalido(msg):
                try:
                    estado_lbl.config(text=msg, fg=SUBTEXT)
                except Exception:
                    pass
                self._form_listo = False
                self._draw_scan_btn(False)
                _draw_guardar(False)

            if not nombre or not ap_pat:
                _set_invalido("Escribe nombre y apellido paterno");  return
            ok, msg = validar_numero_cuenta(cuenta)
            if not ok:
                _set_invalido(f"Cuenta: {msg}");  return
            if rol in ("admin", "maestro"):
                ok2, msg2 = validar_correo(correo)
                if not ok2:
                    _set_invalido(f"Correo: {msg2}");  return
                ok3, msg3 = validar_contrasena(pwd)
                if not ok3:
                    _set_invalido(f"Contraseña: {msg3}");  return
            else:
                ok4, msg4 = validar_grado(grado)
                if not ok4:
                    _set_invalido(f"Grado: {msg4}");  return
                ok5, msg5 = validar_grupo(grupo)
                if not ok5:
                    _set_invalido(f"Grupo: {msg5}");  return
            # Todo válido
            try:
                estado_lbl.config(
                    text="✓  Formulario completo — presiona GUARDAR",
                    fg=SUCCESS)
            except Exception:
                pass
            self._form_listo = True
            self._draw_scan_btn(True)
            _draw_guardar(True)

        # Guardar datos del alumno cuando cambian (para el panel inferior)
        def _guardar_datos(*args):
            nombre = self.nombre_var.get().strip()
            ap_pat = self.ap_pat_var.get().strip()
            ap_mat = self.ap_mat_var.get().strip()
            cuenta = self.cuenta_var.get().strip()
            rol    = self.rol_var.get()
            grado  = self.grado_var.get()
            grupo  = self.grupo_var.get()
            nc = f"{ap_pat}, {nombre} {ap_mat}".strip(", ").strip()
            try:
                self.reg_nombre_var.set(nc)
                gd = f" · Grado {grado} · Grupo {grupo}" if rol == "estudiante" else ""
                self.reg_detalle_var.set(f"Cuenta: {cuenta} · {rol.capitalize()}{gd}")
            except Exception:
                pass
            _validar_y_actualizar()

        for v in (self.nombre_var, self.ap_pat_var, self.ap_mat_var,
                  self.cuenta_var, self.correo_var, self.pwd_var):
            v.trace_add("write", _guardar_datos)
        self.rol_var.trace_add("write", _guardar_datos)
        self.grado_var.trace_add("write", _guardar_datos)
        self.grupo_var.trace_add("write", _guardar_datos)

        for i, rol in enumerate(ROLES_VALIDOS):
            c = COLOR_ROL.get(rol, ACCENT)
            tk.Radiobutton(rol_frame, text=rol.capitalize(),
                           variable=self.rol_var, value=rol,
                           font=(FONT, 9, "bold"), fg=c, bg="#E9EEF6",
                           selectcolor="#D7F5E7", activebackground="#E9EEF6",
                           activeforeground=c, cursor="hand2",
                           command=lambda: (self._actualizar_campos_rol(),
                                            _guardar_datos())
                           ).place(x=16 + i * 122, y=10)

        # Zona condicional
        self._frame_cond = tk.Frame(sheet, bg="#F8FAFC", width=W - 36, height=66)
        self._frame_cond.place(x=PAD, y=370)
        self._frame_cond.pack_propagate(False)

        self._lbl_correo = tk.Label(self._frame_cond, text="Correo",
                                    font=(FONT, 9, "bold"), fg="#344054", bg="#F8FAFC")
        self._ent_correo = tk.Entry(self._frame_cond, textvariable=self.correo_var,
                                    font=(FONT, 10), fg=TEXT, bg=CARD,
                                    insertbackground=ACCENT, relief="flat",
                                    highlightthickness=1, highlightcolor=ACCENT,
                                    highlightbackground=BORDER)
        self._lbl_pwd = tk.Label(self._frame_cond, text="Contraseña",
                                  font=(FONT, 9, "bold"), fg="#344054", bg="#F8FAFC")
        self._ent_pwd = tk.Entry(self._frame_cond, textvariable=self.pwd_var,
                                  show="●", font=(FONT, 10), fg=TEXT, bg=CARD,
                                  insertbackground=ACCENT, relief="flat",
                                  highlightthickness=1, highlightcolor=ACCENT,
                                  highlightbackground=BORDER)
        self._lbl_grado = tk.Label(self._frame_cond, text="Grado",
                                    font=(FONT, 9, "bold"), fg="#344054", bg="#F8FAFC")
        self._om_grado  = tk.OptionMenu(self._frame_cond, self.grado_var, *GRADOS)
        self._om_grado.config(font=(FONT, 9), fg=TEXT, bg=CARD,
                               activebackground=CARD, activeforeground=ACCENT,
                               highlightthickness=0, relief="flat", bd=0)
        self._om_grado["menu"].config(bg=CARD, fg=TEXT, font=(FONT, 9))
        self._lbl_grupo = tk.Label(self._frame_cond, text="Grupo",
                                    font=(FONT, 9, "bold"), fg="#344054", bg="#F8FAFC")
        self._om_grupo  = tk.OptionMenu(self._frame_cond, self.grupo_var, *GRUPOS)
        self._om_grupo.config(font=(FONT, 9), fg=TEXT, bg=CARD,
                               activebackground=CARD, activeforeground=ACCENT,
                               highlightthickness=0, relief="flat", bd=0)
        self._om_grupo["menu"].config(bg=CARD, fg=TEXT, font=(FONT, 9))

        # También entran al teclado virtual los campos condicionales.
        touch_entries.extend([self._ent_correo, self._ent_pwd])
        for ent in touch_entries:
            ent.bind("<FocusIn>", lambda e, ee=ent: setattr(self, "_vk_target", ee), add="+")
            ent.bind("<Button-1>", lambda e, ee=ent: (setattr(self, "_vk_target", ee), ee.focus_set()), add="+")

        self._actualizar_campos_rol()

        # Teclado virtual integrado al formulario.
        KEY_Y = min(486, max(455, SHEET_H - 470))
        KEY_H = max(210, min(330, SHEET_H - KEY_Y - 92))
        self._crear_teclado_virtual(sheet, PAD, KEY_Y, W - 2*PAD, KEY_H, FONT)
        try:
            self._vk_target = touch_entries[0]
            touch_entries[0].focus_set()
        except Exception:
            pass

        # Disparar validación inicial con los datos que ya tenga
        _validar_y_actualizar()

        # Handler del botón GUARDAR: valida una ultima vez, cierra el sheet
        # y deja _form_listo correctamente sincronizado (corrige el bug donde
        # a veces el registro no avanzaba aunque el formulario estuviera listo).
        def _on_guardar_click(e):
            _validar_y_actualizar()    # forzar refresco de _form_listo
            if self._form_listo:
                _close_sheet()
        guardar_cv.bind("<Button-1>", _on_guardar_click)

        # Animar apertura
        def _slide_up(y):
            target = WIN_H - SHEET_H
            if y > target:
                sheet.place(y=y)
                self.after(10, lambda: _slide_up(y - 28))
            else:
                sheet.place(y=target)
        _slide_up(WIN_H)

    def _lanzar_desde_btn(self):
        """Llamado al presionar el botón INICIAR ESCANEO en la cámara."""
        nombre = self.nombre_var.get().strip()
        ap_pat = self.ap_pat_var.get().strip()
        ap_mat = self.ap_mat_var.get().strip()
        cuenta = self.cuenta_var.get().strip()
        correo = self.correo_var.get().strip()
        pwd    = self.pwd_var.get()
        rol    = self.rol_var.get()
        grado  = self.grado_var.get()
        grupo  = self.grupo_var.get()

        # ── Pre-validacion: cuenta duplicada ──────────────────────────────
        # Evita iniciar el escaneo si la cuenta ya esta en la BD.
        # Avisa al usuario antes de que la camara empiece a capturar.
        try:
            conn = conectar()
            c = conn.cursor()
            c.execute(
                "SELECT id, nombre, apellido_paterno FROM usuarios "
                "WHERE numero_cuenta = ?", (cuenta,))
            existe = c.fetchone()
            conn.close()
            if existe:
                nom = f"{existe[1]} {existe[2]}".strip()
                try:
                    self.posicion_var.set(
                        f"✗  Cuenta ya registrada: {nom}")
                    if hasattr(self, "_draw_pill_reg"):
                        self._draw_pill_reg(DANGER)
                    self.reg_detalle_var.set(
                        f"Numero de cuenta {cuenta} ya pertenece a otro usuario")
                except Exception:
                    pass
                return   # NO inicia el escaneo
        except Exception as e:
            print(f"[REG] Error validando duplicado: {e}")

        # Ocultar FAB y botón de escaneo, mostrar cancelar
        try:
            self.scan_btn_cv.place_forget()
            self._fab_cv.place_forget()
            self._cancel_btn.place(x=W - 150, y=24, width=136, height=20)
        except Exception:
            pass

        # Restaurar pildora a su color normal por si tenia un error previo
        try:
            self.posicion_var.set("Posiciónate frente a la cámara")
            if hasattr(self, "_draw_pill_reg"):
                self._draw_pill_reg(ACCENT2)
        except Exception:
            pass

        self.after(0, self._resetear_pasos_ui)
        threading.Thread(
            target=self._capturar_registro,
            args=(nombre, ap_pat, ap_mat,
                  cuenta, correo, pwd, rol, grado, grupo),
            daemon=True).start()


    # ── Validacion e inicio ───────────────────────────────────────────────────
    def _iniciar_registro(self):
        """Reemplazado por _iniciar_registro_sheet — no se usa en el nuevo diseño."""
        pass


    def _capturar_registro(self, nombre, ap_pat, ap_mat,
                            cuenta, correo, pwd, rol, grado, grupo):
        BAR_W = W - 32

        uid = registrar_usuario(
            nombre=nombre,
            apellido_paterno=ap_pat,
            apellido_materno=ap_mat,
            numero_cuenta=cuenta,
            correo=correo if rol in ("admin", "maestro") else None,
            contrasena=pwd if rol in ("admin", "maestro") else None,
            rol=rol,
            grado=int(grado) if rol == "estudiante" else None,
            grupo=grupo.upper() if rol == "estudiante" else None,
            registrado_por=self._usuario_login.get("id")
                           if self._usuario_login else None)

        if uid == -1:
            # ── Error visible al usuario y restauracion de la UI ─────────────
            # La causa mas comun es DUPLICADO: ese numero de cuenta ya existe.
            # El error de la BD se imprime en consola pero hay que avisar
            # tambien en pantalla y dejar al usuario volver a editar.
            def _mostrar_error():
                # 1. Pildora superior roja con mensaje claro
                try:
                    self.posicion_var.set(
                        "✗  Esa cuenta ya esta registrada. Edita el formulario.")
                    if hasattr(self, "_draw_pill_reg"):
                        self._draw_pill_reg(DANGER)
                except Exception:
                    pass
                # 2. Tambien en el panel inferior
                try:
                    self.reg_detalle_var.set(
                        "Numero de cuenta duplicado — verifica los datos")
                except Exception:
                    pass
                # 3. Restaurar boton de escaneo y FAB; ocultar Cancelar
                try:
                    BTN_W, BTN_H_b = W - 32, 42
                    BTN_Y_b = self.cam_label.winfo_y() + self.cam_label.winfo_height() - BTN_H_b - 14
                    self.scan_btn_cv.place(x=16, y=BTN_Y_b)
                    if hasattr(self, "_fab_cv"):
                        BTN_R = 28
                        FAB_H = BTN_R * 2
                        self._fab_cv.place(x=W - FAB_H - 16,
                                           y=BTN_Y_b - FAB_H - 10)
                    if hasattr(self, "_cancel_btn"):
                        self._cancel_btn.place_forget()
                except Exception:
                    pass
                # 4. Marcar formulario como NO listo (forzar nueva edicion)
                self._form_listo = False
                try:
                    self._draw_scan_btn(False)
                except Exception:
                    pass
            self.after(0, lambda: self._safe(_mostrar_error))
            return

        vectores_angulo: dict = {}
        n_muestras_paso = [0] * N_PASOS
        t_offsets = []; acum = 0.0
        for p in PASOS_REGISTRO:
            t_offsets.append(acum); acum += p[4]

        for paso_idx, (_, icono, instruccion, etiqueta,
                       duracion, modo_det,
                       tipo_esperado, msg_correccion) in \
                enumerate(PASOS_REGISTRO):

            vectores_paso = []; t_paso_activo = 0.0
            t_ultimo_tick = None; ultimo_id = -1

            self._modo_deteccion = modo_det
            self._tipo_esperado  = tipo_esperado

            self.after(0, lambda i=instruccion: self._safe(
                lambda: self.status_var.set(i)))
            self.after(0, lambda pi=paso_idx:
                       self._activar_paso_ui(pi, 0.0))
            self._set_overlay((255, 184, 48),
                               f"{paso_idx+1}/{N_PASOS}: {instruccion}")

            while t_paso_activo < duracion and self.cam_running:
                if len(vectores_paso) >= MAX_MUESTRAS_PASO:
                    t_paso_activo = duracion; break

                with self._analisis_lock:
                    frame_id       = self._analisis["frame_id"]
                    v              = self._analisis["vector"]
                    tipo_det       = self._analisis["tipo"]
                    coords_det     = self._analisis["coords"]
                    ocluido        = self._analisis.get("ocluido", False)
                    razon_oclusion = self._analisis.get("razon_oclusion", "")

                angulo_ok     = (tipo_det == tipo_esperado) and (v is not None)
                cara_presente = coords_det is not None
                ahora = time.time()

                if angulo_ok:
                    if t_ultimo_tick is not None:
                        t_paso_activo += ahora - t_ultimo_tick
                    t_ultimo_tick = ahora
                else:
                    t_ultimo_tick = None

                restante     = max(0.0, duracion - t_paso_activo)
                progreso_paso= min(1.0, t_paso_activo / duracion)
                elapsed_g    = t_offsets[paso_idx] + t_paso_activo
                progreso_tot = min(1.0, elapsed_g / TIEMPO_ESCANEO)

                self.after(0, lambda s=int(restante)+1: self._safe(
                    lambda: self.timer_var.set(f"{s}s")))
                self.after(0, lambda pt=int(progreso_tot*BAR_W): self._safe(
                    lambda: self.prog_bar.config(width=pt)))
                self.after(0, lambda pi=paso_idx, pp=progreso_paso:
                           self._activar_paso_ui(pi, pp))

                if frame_id != ultimo_id:
                    ultimo_id = frame_id
                    if angulo_ok:
                        vectores_paso.append(v)
                        n_muestras_paso[paso_idx] += 1
                        if tipo_esperado not in vectores_angulo:
                            vectores_angulo[tipo_esperado] = {"vectores": []}
                        vectores_angulo[tipo_esperado]["vectores"].append(v)
                        n   = len(vectores_paso)
                        tot = sum(n_muestras_paso)
                        self._set_overlay((0, 255, 136),
                            f"{paso_idx+1}/{N_PASOS} {etiqueta} "
                            f"[{n}/{MAX_MUESTRAS_PASO}]")
                        self.after(0, lambda nt=tot: self._safe(
                            lambda: self.progreso_var.set(
                                f"Total: {nt} muestras")))
                        self.after(0, lambda: self._safe(
                            lambda: self.prog_label.config(fg=SUCCESS)))
                        self.after(0,
                            lambda pi=paso_idx,
                                   nm=n_muestras_paso[paso_idx]:
                            self._update_barra_paso(pi, nm))
                    elif ocluido:
                        _oc_msgs = {
                            "mascara":     "Descubre tu rostro",
                            "gorra":       "Descubre tu rostro",
                            "mano":        "Descubre tu rostro",
                            "lentes":      "Descubre tu rostro",
                            "obstruccion": "Descubre tu rostro",
                        }
                        msg_oc = _oc_msgs.get(razon_oclusion, "Descubre tu rostro")
                        self._set_overlay((255, 130, 0), msg_oc)
                        self.after(0, lambda m=msg_oc: self._safe(
                            lambda: self.status_var.set(m)))
                        self.after(0, lambda: self._safe(
                            lambda: self.prog_label.config(fg=WARNING)))
                    elif cara_presente and not angulo_ok:
                        self._set_overlay((255, 59, 92),
                                          f"ESPERANDO — {msg_correccion}")
                        self.after(0, lambda mc=msg_correccion: self._safe(
                            lambda: self.status_var.set(mc)))
                        self.after(0, lambda: self._safe(
                            lambda: self.prog_label.config(fg=DANGER)))
                    else:
                        self._set_overlay((255, 184, 48),
                            f"{paso_idx+1}/{N_PASOS}: {instruccion}")
                        self.after(0, lambda i=instruccion: self._safe(
                            lambda: self.status_var.set(i)))
                        self.after(0, lambda: self._safe(
                            lambda: self.prog_label.config(fg=WARNING)))

                time.sleep(0.04)

            # Verificacion anti-duplicado al terminar cada paso
            if self.cam_running and \
                    n_muestras_paso[paso_idx] >= MUESTRAS_MIN_PASO:
                snapshot = {
                    ang: np.mean(datos["vectores"], axis=0).astype(np.float32)
                    for ang, datos in vectores_angulo.items()
                    if datos["vectores"]
                }
                duplicado = verificar_duplicado_facial(snapshot, excluir_id=uid)
                if duplicado:
                    try: eliminar_persona(uid)
                    except: pass
                    self.after(0, lambda d=duplicado:
                               self._cancelar_por_duplicado(d))
                    return

            # Beep de fin de paso (sin LEDs, distinto al de acceso/denegado)
            threading.Thread(target=servo.beep_registro, daemon=True).start()

            self.after(0, lambda pi=paso_idx:
                       self._activar_paso_ui(pi+1, 0.0))

        # ── Fin escaneo ───────────────────────────────────────────────────────
        self._modo_deteccion = "auto"; self._tipo_esperado = None
        self._set_overlay(None, "")
        self.after(0, lambda: self._safe(lambda: self.timer_var.set("")))
        self.after(0, lambda: self._safe(lambda: self.paso_txt_var.set("")))
        self.after(0, lambda: self._safe(
            lambda: self.prog_bar.config(width=BAR_W)))

        total_muestras = sum(n_muestras_paso)
        pasos_ok       = sum(1 for n in n_muestras_paso
                             if n >= MUESTRAS_MIN_PASO)
        nombre_completo = f"{nombre} {ap_pat} {ap_mat}".strip()

        if pasos_ok == N_PASOS:
            snapshot_final = {
                ang: np.mean(datos["vectores"], axis=0).astype(np.float32)
                for ang, datos in vectores_angulo.items()
                if datos["vectores"]
            }
            duplicado = verificar_duplicado_facial(snapshot_final, excluir_id=uid)
            if duplicado:
                try: eliminar_persona(uid)
                except: pass
                self.after(0, lambda d=duplicado:
                           self._cancelar_por_duplicado(d))
                return

            guardar_vectores_por_angulo(uid, vectores_angulo)
            self.after(0, lambda: self._safe(
                lambda: self.prog_bar.config(bg=SUCCESS)))
            self.after(0, lambda: self._safe(
                lambda: self.status_var.set(
                    f"Registro completo — {total_muestras} muestras")))
            self.after(0, lambda nv=nombre_completo,
                              nt=total_muestras: self._safe(
                lambda: self.progreso_var.set(
                    f"Listo. {nv}\n{nt} muestras.")))
            self.after(0, lambda: self._safe(
                lambda: self.prog_label.config(fg=SUCCESS)))
            self.after(0, lambda: self._safe(
                lambda: None))  # cap_btn removed
            for var in (self.nombre_var, self.ap_pat_var, self.ap_mat_var,
                        self.cuenta_var, self.correo_var, self.pwd_var):
                self.after(0, lambda v=var: self._safe(lambda: v.set("")))
            self.after(0, lambda: self._safe(
                lambda: self.rol_var.set("estudiante")))
            self.after(0, lambda: self._safe(
                lambda: self.grado_var.set("1")))
            self.after(0, lambda: self._safe(
                lambda: self.grupo_var.set("A")))

            # Registro terminado: preguntar si se continuará registrando
            # o si se regresa a la pantalla principal de ACCESO.
            self.after(1200, lambda nv=nombre_completo, nt=total_muestras:
                       self._mostrar_dialogo_post_registro(nv, nt))
        else:
            try: eliminar_persona(uid)
            except: pass
            pasos_fallidos = [PASOS_REGISTRO[i][3]
                              for i, n in enumerate(n_muestras_paso)
                              if n < MUESTRAS_MIN_PASO]
            self.after(0, lambda pf=pasos_fallidos: self._safe(
                lambda: self.progreso_var.set(
                    f"Pasos incompletos:\n{', '.join(pf)}")))
            self.after(0, lambda: self._safe(
                lambda: self.prog_label.config(fg=DANGER)))
            self.after(0, lambda: self._safe(
                lambda: self.status_var.set(
                    "Intentalo de nuevo.")))
            self.after(0, lambda: self._safe(
                lambda: None))  # cap_btn removed in new design
            self.after(0, self._resetear_pasos_ui)

    def _mostrar_dialogo_post_registro(self, nombre="", total_muestras=0):
        """
        Pantalla de decisión al terminar un registro exitoso.
        Sí  → vuelve a Registro para capturar otro usuario.
        No  → regresa automáticamente a Acceso.
        """
        try:
            self._stop_cam()
        except Exception:
            pass

        FONT = "Segoe UI"
        win_h = self.winfo_height() or H

        overlay = tk.Frame(self, bg="#0A0E1A")
        overlay.place(x=0, y=0, width=W, height=win_h)
        overlay.lift()

        cv = tk.Canvas(overlay, width=W, height=win_h,
                       bg="#0A0E1A", highlightthickness=0)
        cv.place(x=0, y=0)

        # Fondo con acentos institucionales, ligero para no cargar el código.
        cv.create_rectangle(0, 0, W, win_h, fill="#0A0E1A", outline="")
        cv.create_oval(-90, -80, 220, 220, fill="#102B3E", outline="")
        cv.create_oval(W - 170, win_h - 180, W + 120, win_h + 130,
                       fill="#12351F", outline="")
        cv.create_rectangle(0, 0, W, 5, fill=TEAL_LN, outline="")
        cv.create_rectangle(0, 5, W, 8, fill=GOLD_LN, outline="")

        card_w = min(W - 44, 420)
        card_h = 390
        card_x = (W - card_w) // 2
        card_y = max(42, (win_h - card_h) // 2)

        # Sombra + tarjeta.
        _round_rect(cv, card_x + 5, card_y + 7,
                    card_x + card_w + 5, card_y + card_h + 7,
                    r=24, fill="#000000", outline="#000000", width=0)
        _round_rect(cv, card_x, card_y, card_x + card_w, card_y + card_h,
                    r=24, fill="#FFFFFF", outline=BORDER, width=2)

        # Banda superior.
        _round_rect(cv, card_x, card_y, card_x + card_w, card_y + 72,
                    r=24, fill=NAVY_LN, outline=NAVY_LN, width=0)
        cv.create_rectangle(card_x, card_y + 48, card_x + card_w,
                            card_y + 72, fill=NAVY_LN, outline="")
        cv.create_polygon(card_x, card_y + 72,
                          card_x + 150, card_y + 72,
                          card_x + 132, card_y + 92,
                          card_x, card_y + 92,
                          fill=TEAL_LN, outline="")
        cv.create_rectangle(card_x, card_y + 92, card_x + card_w,
                            card_y + 96, fill=GOLD_LN, outline="")
        cv.create_text(card_x + card_w // 2, card_y + 35,
                       text="REGISTRO COMPLETADO",
                       font=(FONT, 15, "bold"), fill="#FFFFFF")

        # Icono de éxito.
        icx = card_x + card_w // 2
        icy = card_y + 138
        cv.create_oval(icx - 38, icy - 38, icx + 38, icy + 38,
                       fill="#E6F6EE", outline=ACCENT, width=2)
        cv.create_line(icx - 17, icy + 2, icx - 5, icy + 16,
                       fill=ACCENT, width=5)
        cv.create_line(icx - 5, icy + 16, icx + 22, icy - 19,
                       fill=ACCENT, width=5)

        nombre = (nombre or "Usuario").strip()
        nombre_mostrar = nombre if len(nombre) <= 34 else nombre[:31] + "..."
        cv.create_text(card_x + card_w // 2, card_y + 196,
                       text=nombre_mostrar,
                       font=(FONT, 13, "bold"), fill=TEXT)
        cv.create_text(card_x + card_w // 2, card_y + 222,
                       text=f"Se guardaron {total_muestras} muestras biométricas.",
                       font=(FONT, 9), fill=SUBTEXT)
        cv.create_text(card_x + card_w // 2, card_y + 258,
                       text="¿Deseas registrar otro usuario?",
                       font=(FONT, 12, "bold"), fill=NAVY_LN)

        def _continuar():
            try:
                overlay.destroy()
            except Exception:
                pass
            self._show_registro()

        def _ir_acceso():
            try:
                overlay.destroy()
            except Exception:
                pass
            self._show_acceso()

        btn_w = card_w - 58
        btn_h = 44
        bx = card_x + 29
        by1 = card_y + 286
        by2 = by1 + 54

        btn_si = tk.Button(
            overlay, text="Sí, registrar otro usuario",
            font=(FONT, 10, "bold"), fg="#FFFFFF", bg=ACCENT,
            activebackground=ACCENT2, activeforeground="#FFFFFF",
            relief="flat", bd=0, cursor="hand2",
            command=_continuar)
        btn_si.place(x=bx, y=by1, width=btn_w, height=btn_h)

        btn_no = tk.Button(
            overlay, text="No, volver a Acceso",
            font=(FONT, 10, "bold"), fg=NAVY_LN, bg="#EEF2F7",
            activebackground="#DDE7F0", activeforeground=NAVY_LN,
            relief="flat", bd=0, cursor="hand2",
            command=_ir_acceso)
        btn_no.place(x=bx, y=by2, width=btn_w, height=btn_h)

        # Enter = continuar; Escape = volver a Acceso.
        overlay.bind("<Return>", lambda e: _continuar())
        overlay.bind("<Escape>", lambda e: _ir_acceso())
        btn_si.focus_set()


    # ══════════════════════════════════════════════════════════════════════════
    #  DIÁLOGO DE AUTENTICACIÓN PARA MENÚ
    # ══════════════════════════════════════════════════════════════════════════
    def _pedir_auth_menu(self, on_success=None):
        FONT  = "Segoe UI"
        BEIGE = "#F5E6C8"
        GREEN = "#1A7A4A"
        GREEN_HOV = "#23A062"
        KB_BG = "#EAF0F7"

        overlay = tk.Frame(self, bg="#0A0E1A")
        overlay.place(x=0, y=0, width=W, height=self.winfo_height())
        overlay.lift()

        # Más ancho para que el teclado táctil quepa cómodo en pantalla vertical.
        CARD_W = min(W - 28, 520)
        CARD_X = (W - CARD_W) // 2
        CARD_Y = max(12, (self.winfo_height() - 650) // 2)

        card = tk.Frame(overlay, bg="#FFFFFF",
                        highlightthickness=2,
                        highlightbackground=BEIGE, bd=0)
        card.place(x=CARD_X, y=CARD_Y, width=CARD_W)

        tk.Frame(card, bg=BEIGE, height=6).pack(fill="x")

        ico_cv = tk.Canvas(card, width=46, height=46,
                           bg="#FFFFFF", highlightthickness=0)
        ico_cv.pack(pady=(14, 0))
        ico_cv.create_rectangle(9, 22, 37, 41, fill=NAVY_LN, outline="", width=0)
        ico_cv.create_arc(13, 6, 33, 30, start=0, extent=180,
                          outline=NAVY_LN, style="arc", width=4)
        ico_cv.create_oval(20, 28, 26, 34, fill="#FFFFFF", outline="")
        ico_cv.create_rectangle(22, 31, 24, 38, fill="#FFFFFF", outline="")

        tk.Label(card, text="Acceso Restringido",
                 font=(FONT, 13, "bold"), fg=NAVY_LN,
                 bg="#FFFFFF").pack(pady=(8, 2))
        tk.Label(card, text="Solo administradores y maestros",
                 font=(FONT, 8), fg="#777777",
                 bg="#FFFFFF").pack(pady=(0, 10))

        tk.Frame(card, bg=BEIGE, height=1).pack(fill="x")

        fields_bg = "#FAFAF7"
        fields = tk.Frame(card, bg=fields_bg)
        fields.pack(fill="x")

        auth_cuenta_var = tk.StringVar()
        auth_pwd_var    = tk.StringVar()
        self._auth_vk_target = None
        self._auth_vk_shift  = False

        def _mk_field(parent, label_text, var, show=None):
            tk.Label(parent, text=label_text,
                     font=(FONT, 8), fg="#555555",
                     bg=fields_bg, anchor="w").pack(
                         fill="x", padx=24, pady=(10, 2))
            ent = tk.Entry(parent, textvariable=var,
                           font=(FONT, 11), fg=NAVY_LN, bg="#FFFFFF",
                           insertbackground=GREEN, relief="flat",
                           highlightthickness=1,
                           highlightcolor=GREEN,
                           highlightbackground="#CCCCCC")
            if show:
                ent.config(show=show)
            ent.pack(fill="x", padx=24, ipady=6)

            def _set_target(_ev=None, ee=ent):
                self._auth_vk_target = ee
                try:
                    ee.focus_set()
                except Exception:
                    pass
            ent.bind("<FocusIn>", _set_target)
            ent.bind("<Button-1>", _set_target)
            return ent

        ent_cuenta = _mk_field(fields, "Número de cuenta", auth_cuenta_var)
        ent_pwd    = _mk_field(fields, "Contraseña",       auth_pwd_var, show="●")

        error_var = tk.StringVar(value="")
        tk.Label(fields, textvariable=error_var,
                 font=(FONT, 8), fg="#CC2222",
                 bg=fields_bg).pack(pady=(6, 0))

        # ── Teclado táctil para matrícula/cuenta y contraseña ────────────────
        # Se dibuja dentro del propio diálogo para evitar depender del teclado
        # del sistema operativo de Raspberry Pi OS.
        kb = tk.Frame(fields, bg=KB_BG, height=226)
        kb.pack(fill="x", padx=14, pady=(8, 8))
        kb.pack_propagate(False)

        kb_head = tk.Frame(kb, bg=KB_BG, height=24)
        kb_head.pack(fill="x")
        tk.Label(kb_head, text="⌨  Teclado táctil",
                 font=(FONT, 8, "bold"), fg=NAVY_LN,
                 bg=KB_BG, anchor="w").pack(side="left", padx=8)
        tk.Label(kb_head, text="Toca un campo y escribe aquí",
                 font=(FONT, 7), fg=SUBTEXT,
                 bg=KB_BG, anchor="e").pack(side="right", padx=8)

        kb_rows = tk.Frame(kb, bg=KB_BG)
        kb_rows.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        def _auth_target():
            ent = getattr(self, "_auth_vk_target", None)
            try:
                if ent is not None and ent.winfo_exists():
                    ent.focus_set()
                    return ent
            except Exception:
                pass
            return None

        def _auth_insert(txt):
            ent = _auth_target()
            if ent is None:
                return
            try:
                ent.insert(ent.index(tk.INSERT), txt)
            except Exception:
                pass

        def _auth_backspace():
            ent = _auth_target()
            if ent is None:
                return
            try:
                pos = ent.index(tk.INSERT)
                if pos > 0:
                    ent.delete(pos - 1, pos)
            except Exception:
                pass

        def _auth_clear():
            ent = _auth_target()
            if ent is None:
                return
            try:
                ent.delete(0, tk.END)
            except Exception:
                pass

        def _auth_press(key):
            if key == "⌫":
                _auth_backspace()
            elif key == "Limpiar":
                _auth_clear()
            elif key == "Espacio":
                _auth_insert(" ")
            elif key == "⇧":
                self._auth_vk_shift = not getattr(self, "_auth_vk_shift", False)
                _draw_auth_keys()
            else:
                if len(key) == 1 and key.isalpha():
                    _auth_insert(key.upper() if getattr(self, "_auth_vk_shift", False) else key.lower())
                else:
                    _auth_insert(key)

        def _mk_key(row, text, bg="#FFFFFF", fg=TEXT):
            b = tk.Button(row, text=text,
                          font=(FONT, 8, "bold"), fg=fg, bg=bg,
                          activebackground="#DCEFE7",
                          activeforeground=TEXT,
                          relief="flat", bd=0,
                          highlightthickness=1,
                          highlightbackground="#CFD8E6",
                          cursor="hand2",
                          command=lambda k=text: _auth_press(k))
            b.pack(side="left", fill="both", expand=True, padx=2, pady=2)
            return b

        def _draw_auth_keys():
            for child in kb_rows.winfo_children():
                child.destroy()
            layouts = [
                list("1234567890"),
                list("qwertyuiop"),
                list("asdfghjklñ"),
                ["⇧"] + list("zxcvbnm") + ["⌫"],
                ["@", "_", "-", ".", "/", "Espacio", ".com", "Limpiar"],
            ]
            for layout in layouts:
                row = tk.Frame(kb_rows, bg=KB_BG)
                row.pack(fill="x", expand=True)
                for key in layout:
                    show = key
                    if len(key) == 1 and key.isalpha():
                        show = key.upper() if getattr(self, "_auth_vk_shift", False) else key.lower()
                    if key == "⇧":
                        active = getattr(self, "_auth_vk_shift", False)
                        _mk_key(row, show,
                                bg=GREEN if active else "#FFFFFF",
                                fg="#FFFFFF" if active else TEXT)
                    elif key in ("⌫", "Limpiar"):
                        _mk_key(row, show, bg="#FFECEC", fg=DANGER)
                    elif key in ("Espacio", ".com"):
                        _mk_key(row, show, bg="#EAF7F0", fg=GREEN)
                    else:
                        _mk_key(row, show)

        _draw_auth_keys()
        self._auth_vk_target = ent_cuenta
        ent_cuenta.focus_set()

        tk.Frame(card, bg=BEIGE, height=1).pack(fill="x")

        btn_row = tk.Frame(card, bg="#FFFFFF")
        btn_row.pack(fill="x")

        def _cerrar():
            overlay.destroy()

        def _confirmar():
            cuenta = auth_cuenta_var.get().strip()
            pwd    = auth_pwd_var.get().strip()
            if not cuenta or not pwd:
                error_var.set("Completa ambos campos.")
                return

            # ── Validación real contra la base de datos ────────────────
            try:
                conn = conectar()
                c = conn.cursor()
                if '@' in cuenta:
                    c.execute("""
                        SELECT id, nombre, apellido_paterno, apellido_materno,
                               numero_cuenta, correo, rol, contrasena
                        FROM usuarios
                        WHERE LOWER(correo)=LOWER(?) AND rol IN ('admin','maestro')
                    """, (cuenta,))
                else:
                    c.execute("""
                        SELECT id, nombre, apellido_paterno, apellido_materno,
                               numero_cuenta, correo, rol, contrasena
                        FROM usuarios
                        WHERE numero_cuenta=? AND rol IN ('admin','maestro')
                    """, (cuenta,))
                user = c.fetchone()
                conn.close()

                if not user:
                    error_var.set("Usuario no encontrado o sin permisos.")
                    return

                hash_input = hashlib.sha256(pwd.encode()).hexdigest()
                if hash_input != user[7]:
                    error_var.set("Contraseña incorrecta.")
                    return

                usuario_data = {
                    'id':              user[0],
                    'nombre':          user[1],
                    'apellido_paterno':user[2],
                    'apellido_materno':user[3],
                    'numero_cuenta':   user[4],
                    'correo':          user[5],
                    'rol':             user[6]
                }

                overlay.destroy()
                self._stop_cam()
                self._usuario_login = usuario_data

                if on_success:
                    on_success(usuario_data)   # ← ejecuta la acción personalizada
                else:
                    self._build_main()          # ← comportamiento original
            except Exception as e:
                error_var.set(f"Error de conexión: {e}")

        btn_cancel = tk.Button(
            btn_row, text="Cancelar",
            font=(FONT, 10), fg="#555555", bg="#FFFFFF",
            relief="flat", cursor="hand2", bd=0,
            highlightthickness=0, command=_cerrar)
        btn_cancel.pack(side="left", fill="both", expand=True, ipady=12)
        btn_cancel.bind("<Enter>", lambda e: btn_cancel.config(bg="#F0EDE6"))
        btn_cancel.bind("<Leave>", lambda e: btn_cancel.config(bg="#FFFFFF"))

        tk.Frame(btn_row, bg=BEIGE, width=1).pack(side="left", fill="y")

        btn_ok = tk.Button(
            btn_row, text="Ingresar  ▶",
            font=(FONT, 10, "bold"), fg="#FFFFFF", bg=GREEN,
            relief="flat", cursor="hand2", bd=0,
            highlightthickness=0, command=_confirmar)
        btn_ok.pack(side="left", fill="both", expand=True, ipady=12)
        btn_ok.bind("<Enter>", lambda e: btn_ok.config(bg=GREEN_HOV))
        btn_ok.bind("<Leave>", lambda e: btn_ok.config(bg=GREEN))

        ent_cuenta.bind("<Return>", lambda e: ent_pwd.focus_set())
        ent_pwd.bind("<Return>",    lambda e: _confirmar())
        overlay.bind("<Escape>",    lambda e: _cerrar())

    # ══════════════════════════════════════════════════════════════════════════
    #  PANTALLA ACCESO
    # ══════════════════════════════════════════════════════════════════════════
    def _show_acceso(self):
        # Asegurar que cualquier cámara previa esté liberada
        self._stop_cam()
        self._clear()
        # Adaptar a la altura real de pantalla para que el panel inferior siempre sea visible
        SCREEN_H = self.winfo_screenheight()
        WIN_H    = min(H, SCREEN_H)
        self.geometry(f"{W}x{WIN_H}+0+0")
        self.verificando = False
        self._modo_acceso = True
        self._set_overlay(None, "")

        FONT   = "Segoe UI"
        HDR_H  = 56
        PILL_H = 36
        PILL_Y = HDR_H + 10
        CAM_Y  = PILL_Y + PILL_H + 10
        BOT_H  = 130
        CAM_H  = WIN_H - CAM_Y - BOT_H   # cámara ocupa exactamente el espacio central

        # ── Header verde ──────────────────────────────────────────────────────
        hdr = tk.Canvas(self, width=W, height=HDR_H, bg=ACCENT, highlightthickness=0)
        hdr.place(x=0, y=0)
        hdr.create_rectangle(0, 0, W, HDR_H, fill=ACCENT, outline="")
        hdr.create_text(W // 2, HDR_H // 2 - 5,
                        text="ACCESO", font=(FONT, 14, "bold"),
                        fill="#FFFFFF", anchor="center")

        # ── Botón MENÚ (lleva al main con autenticación) ──────────────────────
        menu_btn = tk.Button(
            self, text="☰  Menú",
            font=(FONT, 8, "bold"), fg="#FFFFFF", bg=NAVY_LN,
            relief="flat", cursor="hand2", bd=0,
            highlightthickness=1, highlightbackground=TEAL_LN,
            command=self._pedir_auth_menu)
        menu_btn.place(x=W - 90, y=HDR_H // 2 - 12, width=80, height=26)
        menu_btn.bind("<Enter>", lambda e: menu_btn.config(bg="#1A6FD4", fg="#FFFFFF"))
        menu_btn.bind("<Leave>", lambda e: menu_btn.config(bg=NAVY_LN,  fg="#FFFFFF"))

        self.hdr_status_var = tk.StringVar(value="")
        self.hdr_status_lbl = tk.Label(
            self, textvariable=self.hdr_status_var,
            font=(FONT, 9, "bold"), fg="#AAFFAA", bg=ACCENT,
            anchor="center", justify="center")
        self.hdr_status_lbl.place(x=90, y=HDR_H // 2 - 10, width=W - 190, height=20)

        self.hdr_nombre_var = tk.StringVar(value="")  # se mantiene por compatibilidad

        # ── Franja beige entre header y cámara ────────────────────────────────
        tk.Frame(self, bg=BG, width=W, height=PILL_H + 20).place(x=0, y=HDR_H)

        # ── Pill de estado ────────────────────────────────────────────────────
        PILL_W = 240
        self.pill_cv = tk.Canvas(self, width=PILL_W, height=PILL_H,
                                 bg=BG, highlightthickness=0)
        self.pill_cv.place(x=W // 2 - PILL_W // 2, y=PILL_Y)

        self.posicion_var = tk.StringVar(value="Esperando...")

        def _draw_pill(color):
            r = PILL_H // 2
            self.pill_cv.delete("all")
            self.pill_cv.create_arc(0, 0, 2*r, PILL_H,
                                    start=90, extent=180,
                                    fill=color, outline=color)
            self.pill_cv.create_arc(PILL_W - 2*r, 0, PILL_W, PILL_H,
                                    start=270, extent=180,
                                    fill=color, outline=color)
            self.pill_cv.create_rectangle(r, 0, PILL_W - r, PILL_H,
                                          fill=color, outline="")
            try:
                self.posicion_label.config(bg=color)
            except Exception:
                pass

        _draw_pill(ACCENT2)
        self._draw_pill = _draw_pill

        self.posicion_label = tk.Label(
            self.pill_cv, textvariable=self.posicion_var,
            font=(FONT, 9, "bold"), fg="#FFFFFF",
            bg=ACCENT2, wraplength=PILL_W - 16)
        self.posicion_label.place(relx=.5, rely=.5, anchor="center")

        # ── Zona cámara con fondo visual institucional ─────────────────────────
        self._crear_fondo_camara(0, CAM_Y, W, CAM_H,
                                 "Verificación de acceso",
                                 "Mira directo a la cámara para identificarte")
        self.cam_label = tk.Label(self, bg="#0B1120", bd=0, highlightthickness=0)
        self.cam_label.place(x=8, y=CAM_Y + 8, width=W - 16, height=CAM_H - 16)

        # ── Panel inferior oscuro ─────────────────────────────────────────────
        bot = tk.Frame(self, bg=NAVY_LN, width=W, height=BOT_H)
        bot.place(x=0, y=WIN_H - BOT_H)
        bot.pack_propagate(False)

        # Línea separadora superior (acento verde)
        tk.Frame(bot, bg=TEAL_LN, width=W, height=3).place(x=0, y=0)

        # Nombre del alumno
        self.candidato_var = tk.StringVar(value="")
        tk.Label(bot, textvariable=self.candidato_var,
                 font=(FONT, 12, "bold"), fg="#FFFFFF", bg=NAVY_LN,
                 anchor="w").place(x=14, y=8, width=W - 28)

        # Detalle: cuenta · rol · ángulo
        self.detalle_var = tk.StringVar(value="")
        self.detalle_lbl = tk.Label(
            bot, textvariable=self.detalle_var,
            font=(FONT, 8), fg="#AABBCC", bg=NAVY_LN, anchor="w")
        self.detalle_lbl.place(x=14, y=34, width=W - 28)

        # Etiqueta SIMILITUD + porcentaje (en la misma línea)
        tk.Label(bot, text="SIMILITUD", font=(FONT, 7, "bold"),
                 fg="#5577AA", bg=NAVY_LN, anchor="w"
                 ).place(x=14, y=62)
        self.sim_lbl = tk.Label(bot, text="",
                                font=(FONT, 9, "bold"),
                                fg="#FFFFFF", bg=NAVY_LN, anchor="e")
        self.sim_lbl.place(x=W - 60, y=58, width=46)

        # Barra de similitud con fondo oscuro y relleno coloreado
        BAR_W  = W - 28
        BAR_H  = 16
        BAR_Y  = 82
        BAR_R  = 8   # radio para efecto redondeado con Canvas

        sim_bg = tk.Canvas(bot, width=BAR_W, height=BAR_H,
                           bg="#0D1F38", highlightthickness=0)
        sim_bg.place(x=14, y=BAR_Y)
        # Fondo de la barra con esquinas redondeadas
        sim_bg.create_arc(0,          0, BAR_R*2,   BAR_H,   start=90,  extent=180,
                          fill="#0D1F38", outline="#0D1F38")
        sim_bg.create_arc(BAR_W-BAR_R*2, 0, BAR_W, BAR_H,   start=270, extent=180,
                          fill="#0D1F38", outline="#0D1F38")
        sim_bg.create_rectangle(BAR_R, 0, BAR_W-BAR_R, BAR_H,
                                fill="#0D1F38", outline="")
        self._sim_canvas = sim_bg
        self._sim_bar_w  = BAR_W
        self._sim_bar_h  = BAR_H
        self._sim_bar_r  = BAR_R

        # Barra de relleno (se redibuja en _set_sim_bar)
        self.sim_bar = tk.Frame(sim_bg, bg=BORDER, width=0, height=BAR_H)
        self.sim_bar.place(x=0, y=0)

        # Marcadores de referencia (umbral 80% y 90%)
        for pct_mark, label_mark in ((80, "80%"), (90, "90%")):
            mx = int(pct_mark / 100 * BAR_W)
            sim_bg.create_line(mx, 0, mx, BAR_H,
                               fill="#FFFFFF", width=1, dash=(2, 2))

        # Etiqueta de umbrales bajo la barra
        tk.Label(bot, text="0%", font=(FONT, 6), fg="#334455",
                 bg=NAVY_LN).place(x=14, y=BAR_Y + BAR_H + 2)
        tk.Label(bot, text="80%", font=(FONT, 6), fg="#334455",
                 bg=NAVY_LN).place(x=14 + int(0.80 * BAR_W) - 8,
                                   y=BAR_Y + BAR_H + 2)
        tk.Label(bot, text="100%", font=(FONT, 6), fg="#334455",
                 bg=NAVY_LN).place(x=14 + BAR_W - 24, y=BAR_Y + BAR_H + 2)

        # Variables fantasma (compatibilidad con _verificar, _lanzar_verificacion, etc.)
        self.resultado_var   = tk.StringVar(value="")
        self.resultado_label = tk.Label(self, textvariable=self.resultado_var,
                                        font=(FONT, 1), fg=BG, bg=BG)
        self.resultado_label.place(x=-200, y=-200)

        # ── Iniciar cámara y threads ──────────────────────────────────────────
        self._start_cam()
        self.cam_running = True
        self._ultima_cara_t = time.time()
        threading.Thread(target=self._loop_camara,
                         kwargs={"max_w": W - 16, "max_h": CAM_H - 16},
                         daemon=True).start()
        threading.Thread(target=self._loop_analisis, daemon=True).start()
        threading.Thread(target=self._monitor_cara,   daemon=True).start()
        threading.Thread(target=self._guia_posicion,  daemon=True).start()
        self.after(1800, self._lanzar_verificacion)

    def _set_sim_bar(self, pct, color):
        bar_total = W - 28
        w = max(0, min(bar_total, int(pct / 100 * bar_total)))
        try:
            self.sim_lbl.config(text=f"{pct}%" if pct > 0 else "")
        except Exception:
            pass
        # Redibujar barra de relleno en el canvas
        try:
            cv = self._sim_canvas
            bw = self._sim_bar_w
            bh = self._sim_bar_h
            br = self._sim_bar_r
            cv.delete("sim_fill")
            if w > 0:
                r = min(br, w // 2)
                # Arco izquierdo
                cv.create_arc(0, 0, r*2, bh, start=90, extent=180,
                              fill=color, outline=color, tags="sim_fill")
                if w > r:
                    # Arco derecho solo si hay suficiente ancho
                    r2 = min(br, (bw - w) + r)
                    end_x = w
                    cv.create_arc(end_x - r*2, 0, end_x, bh,
                                  start=270, extent=180,
                                  fill=color, outline=color, tags="sim_fill")
                    cv.create_rectangle(r, 0, end_x - r, bh,
                                        fill=color, outline="", tags="sim_fill")
                # Redibujar marcadores encima
                for pct_mark in (80, 90):
                    mx = int(pct_mark / 100 * bw)
                    cv.create_line(mx, 0, mx, bh,
                                   fill="#FFFFFF", width=1, dash=(2, 2),
                                   tags="sim_fill")
        except Exception:
            pass

    def _lanzar_verificacion(self):
        if not self.cam_running: return
        # ── Defensa: solo verificar si REALMENTE estamos en modo acceso ──
        # Un after() programado en acceso podia dispararse despues de que
        # el usuario navego a registro/otro modo. Bloqueamos eso aqui.
        if not getattr(self, "_modo_acceso", False):
            return
        if self.verificando:
            self.after(300, self._lanzar_verificacion); return
        self.verificando = True
        self.after(0, lambda: self.resultado_var.set("Escaneando..."))
        self.after(0, lambda: self.resultado_label.config(fg=ACCENT))
        self.after(0, lambda: self.candidato_var.set(""))
        self.after(0, lambda: self.detalle_var.set(""))
        self._set_overlay((255, 184, 48), "Analizando...")
        threading.Thread(target=self._verificar, daemon=True).start()

    def _verificar(self):
        vectores = []; intentos = 0; ultimo = -1; hay_cara = False
        hay_oclusion = False; razon_oclusion = ""
        while len(vectores) < 8 and intentos < 120 and self.cam_running:
            intentos += 1
            with self._analisis_lock:
                frame_id = self._analisis["frame_id"]
                v        = self._analisis["vector"]
                tipo     = self._analisis["tipo"]
                ocluido  = self._analisis.get("ocluido", False)
                razon_tmp = self._analisis.get("razon_oclusion", "")
            if frame_id == ultimo:
                time.sleep(0.04); continue
            ultimo = frame_id
            if ocluido:
                hay_cara = True
                hay_oclusion = True
                razon_oclusion = razon_tmp or razon_oclusion
            elif v is not None:
                hay_cara = True
                if tipo == TIPO_FRONTAL:
                    vectores.append(v)
            time.sleep(0.04)

        self._set_overlay(None, "")
        self.verificando = False
        if not self.cam_running: return

        if not vectores:
            if hay_oclusion:
                self.after(0, lambda: self.resultado_var.set("ROSTRO CUBIERTO"))
                self.after(0, lambda: self.posicion_var.set("Descubre tu rostro."))
                self.after(0, lambda: self._draw_pill(WARNING))
                self.after(0, lambda: self._set_sim_bar(0, BORDER))
                self._set_overlay((255, 130, 0), "Descubre tu rostro")
                self.after(2500, self._lanzar_verificacion); return

            self.after(0, lambda: self.resultado_var.set(
                "Gira al frente" if hay_cara else "Sin rostro"))
            self.after(0, lambda: self.posicion_var.set(
                "Mira directo a la camara." if hay_cara
                else "Ponte frente a la camara."))
            self.after(0, lambda: self._draw_pill(WARNING))
            self.after(0, lambda: self._set_sim_bar(0, BORDER))
            self.after(3000, self._lanzar_verificacion); return

        v_final   = np.mean(vectores, axis=0).astype(np.float32)
        resultado = reconocer_persona(v_final, angulo_nuevo="frontal")

        if resultado is None:
            from database import cargar_vectores_por_angulo as _cvpa
            hay = len(_cvpa()) > 0
            self.after(0, lambda: self.resultado_var.set(
                "ACCESO DENEGADO" if hay else "Sin registros"))
            self.after(0, lambda: self.candidato_var.set(
                "Persona no reconocida." if hay
                else "No hay usuarios registrados."))
            if hay:
                self.after(0, lambda: self.hdr_status_var.set("DENEGADO ✕"))
                self.after(0, lambda: self.hdr_status_lbl.config(fg="#FF8888"))
                self.after(0, lambda: self.hdr_nombre_var.set(""))
                self.after(0, lambda: self._draw_pill(DANGER))
                self.after(0, lambda: self.posicion_var.set("Acceso denegado"))
            self.after(0, lambda: self._set_sim_bar(0, BORDER))
            if hay:
                self._set_overlay((255, 59, 92), "Desconocido")
                self.after(0, lambda: servo.denegar())
            self.after(4000, self._lanzar_verificacion); return

        sim         = resultado["similitud_pct"]
        color_barra = SUCCESS if resultado["acceso"] else \
                      (WARNING if sim >= 80 else DANGER)
        self.after(0, lambda: self._set_sim_bar(int(sim), color_barra))
        self.after(0, lambda r=resultado: self.detalle_var.set(
            f"Angulo match: {r['angulo']}"))

        if resultado["acceso"] and sim >= 80:
            self.after(0, lambda r=resultado: self._resultado_ok(r))
        else:
            self.after(0, lambda r=resultado: self._resultado_negado(r))

    def _resultado_ok(self, r):
        self._t_acceso_ok = time.time()
        self._set_overlay((0, 255, 136), r["nombre"])
        self.resultado_var.set("ACCESO PERMITIDO")
        self.resultado_label.config(fg=SUCCESS)
        try:
            self.hdr_status_var.set("PERMITIDO ✓")
            self.hdr_status_lbl.config(fg="#AAFFAA")
            self.hdr_nombre_var.set("")
            self._draw_pill(SUCCESS)
            self.posicion_var.set("✓  Listo · mira a la camara")
            rol_txt = r.get("rol", "").capitalize()
            self.candidato_var.set(r["nombre"])
            self.detalle_var.set(
                f"Cuenta: {r.get('numero_cuenta','---')} · {rol_txt}")
            self.detalle_lbl.config(fg="#AAFFAA")
        except Exception:
            pass
        servo.abrir(r["nombre"])
        self.after(4000, self._lanzar_verificacion)

    def _resultado_negado(self, r):
    # Ignorar si el acceso fue concedido hace menos de 5 segundos
        if time.time() - self._t_acceso_ok < 5:
            self.after(4000, self._lanzar_verificacion)
            return
        self._set_overlay((255, 59, 92), "Denegado")
        self.resultado_var.set("ACCESO DENEGADO")
        self.resultado_label.config(fg=DANGER)
        try:
            self.hdr_status_var.set("DENEGADO ✕")
            self.hdr_status_lbl.config(fg="#FF8888")
            self.hdr_nombre_var.set("")
            self._draw_pill(DANGER)
            self.posicion_var.set("Acceso denegado")
            self.candidato_var.set(r["nombre"])
            rol_txt = r.get("rol", "").capitalize()
            self.detalle_var.set(
                f"Cuenta: {r.get('numero_cuenta','---')} · {rol_txt}")
            self.detalle_lbl.config(fg="#FF8888")
        except Exception:
            pass
        servo.denegar()
        self.after(4000, self._lanzar_verificacion)

    def _guia_posicion(self):
        """Actualiza la instruccion de posicion en tiempo real con debounce."""
        DEBOUNCE_SIN_CARA = 1.2
        INTERVALO         = 0.5
        ultimo_msg   = ""
        t_sin_cara   = None

        while self.cam_running:
            time.sleep(INTERVALO)
            if self.verificando:
                ultimo_msg = ""; t_sin_cara = None; continue
            with self._analisis_lock:
                v    = self._analisis["vector"]
                tipo = self._analisis["tipo"]
            try:
                if v is not None:
                    t_sin_cara = None
                    if tipo == TIPO_FRONTAL:
                        msg = "✓  Listo · mira a la camara"; color = ACCENT2
                    elif tipo in (TIPO_PERFIL_D, TIPO_PERFIL_I):
                        msg = "Estas volteado — mira al frente"; color = WARNING
                    else:
                        msg = "✓  Listo · mira a la camara"; color = ACCENT2
                else:
                    if t_sin_cara is None:
                        t_sin_cara = time.time()
                    if (time.time() - t_sin_cara) >= DEBOUNCE_SIN_CARA:
                        msg = "Acercate a la camara"; color = WARNING
                    else:
                        continue
                if msg != ultimo_msg:
                    ultimo_msg = msg
                    _c = color
                    self.after(0, lambda m=msg, c=_c: (
                        self.posicion_var.set(m), self._draw_pill(c)))
            except Exception:
                pass

    def _monitor_cara(self):
        """
        Re-escanea SOLO cuando la cara desaparece y vuelve a aparecer.
        Movimientos normales mientras la cara sigue detectada no disparan nada.
        """
        TIMEOUT_SIN_CARA  = 1.5
        FRAMES_REAPARECE  = 5
        COOLDOWN          = 4.0

        cara_presente      = False
        contador_reaparece = 0
        t_ultimo_scan      = 0.0

        while self.cam_running:
            with self._analisis_lock:
                v = self._analisis["vector"]

            cooldown_ok = (time.time() - t_ultimo_scan) >= COOLDOWN

            if v is not None:
                self._ultima_cara_t = time.time()

                if not cara_presente:
                    contador_reaparece += 1
                    if contador_reaparece >= FRAMES_REAPARECE:
                        cara_presente      = True
                        contador_reaparece = 0
                        if not self.verificando and cooldown_ok:
                            t_ultimo_scan = time.time()
                            self.after(0, self._resetear_pantalla_acceso)
                            self.after(150, self._lanzar_verificacion)
            else:
                contador_reaparece = 0
                sin_cara = time.time() - self._ultima_cara_t

                if sin_cara >= TIMEOUT_SIN_CARA:
                    if cara_presente:
                        cara_presente = False
                    if not self.verificando:
                        self.after(0, self._resetear_pantalla_acceso)

            time.sleep(0.12)

    def _resetear_pantalla_acceso(self):
        try:
            self.resultado_var.set("Esperando...")
            self.resultado_label.config(fg=ACCENT)
            self.candidato_var.set("")
            self.detalle_var.set("")
            self._set_sim_bar(0, BORDER)
            self._set_overlay(None, "")
            self.hdr_status_var.set("")
            self.hdr_nombre_var.set("")
            self._draw_pill(ACCENT2)
            self.posicion_var.set("Esperando...")
            self.detalle_lbl.config(fg="#AABBCC")
        except:
            pass

    # ── Dashboard y sensores IR ───────────────────────────────────────────────
    def _notificar_dashboard(self):
        """Escribe un archivo de señal para que el dashboard refresque."""
        try:
            import os as _os
            _ruta = _os.path.join(_os.path.dirname(__file__),
                                  '..', 'database', '.refresh_signal')
            with open(_ruta, 'w') as f:
                f.write(str(time.time()))
        except Exception:
            pass

    def _on_entrada_sensor(self, uid):
        """S1→S2 detectado. Solo cuenta si hubo escaneo reciente (< 15s)."""
        if time.time() - self._t_acceso_ok > 15:
            print("[IR] Entrada ignorada — sin escaneo reciente")
            return
        if self.sensores_ir:
            self.sensores_ir.incrementar()
        uid_real = getattr(self, '_ultimo_uid_ok', None)
        if not uid_real:
            print("[IR] Entrada ignorada — uid no disponible")
            return
        try:
            registrar_acceso(uid_real, "entrada", detalle="Sensor IR S1→S2")
            print(f"[IR] Entrada registrada en BD (uid={uid_real})")
        except Exception as e:
            print(f"[IR] Error registrando entrada: {e}")
        self._notificar_dashboard()

    def _on_salida_boton(self):
        """Botón presionado — arma el flag, espera que pase por S2→S1."""
        self._t_boton_salida = time.time()
        print("[IR] Botón presionado — esperando paso por sensores")

    def _on_salida_sensor(self, uid):
        """S2→S1 detectado. Solo cuenta si el botón fue presionado recientemente (< 15s)."""
        t_boton = getattr(self, '_t_boton_salida', 0)
        if time.time() - t_boton > 15:
            print("[IR] Salida ignorada — botón no presionado recientemente")
            return
        self._t_boton_salida = 0
        if self.sensores_ir:
            self.sensores_ir.decrementar()
        try:
            conn = conectar()
            conn.execute("""
                INSERT INTO registro_acceso
                    (usuario_id, nombre, apellido_paterno, apellido_materno,
                    numero_cuenta, rol, tipo_evento, detalle)
                VALUES (NULL, 'Salida', 'Manual', '', NULL,
                        'estudiante', 'salida', 'Botón + Sensor S2→S1')
            """)
            conn.commit()
            conn.close()
            print("[IR] Salida registrada en BD")
        except Exception as e:
            print(f"[IR] Error registrando salida: {e}")
        self._notificar_dashboard()

    def _show_dashboard(self):
        """Abre el diálogo de autenticación y, si es válido, lanza el dashboard."""
        self._pedir_auth_menu(on_success=lambda u: self._abrir_dashboard(u))

    def _abrir_dashboard(self, usuario_data):
        """Cierra la ventana actual y abre el Dashboard con los datos del usuario."""
        import sys, os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dashboard_dir = os.path.join(current_dir, 'dashboard')
        if dashboard_dir not in sys.path:
            sys.path.insert(0, dashboard_dir)
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        try:
            from dashboard import Dashboard
            self.destroy()
            dash = Dashboard(usuario_data)
            dash.run()
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_root = tk.Tk()
            error_root.title("Error")
            tk.Label(error_root, text=f"No se pudo abrir el dashboard:\n{e}").pack()
            tk.Button(error_root, text="Cerrar", command=error_root.destroy).pack()
            error_root.mainloop()

    def on_close(self):
        self._stop_cam()
        servo.desconectar()
        if getattr(self, 'sensor', None):
            self.sensor.detener()
        if getattr(self, 'sensores_ir', None):
            self.sensores_ir.detener()
        self.destroy()


if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("[ERROR] Falta Pillow:  pip install Pillow")
        exit(1)
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()