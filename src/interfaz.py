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
                          TIPO_FRONTAL, TIPO_PERFIL_D, TIPO_PERFIL_I)
from database   import (registrar_usuario, guardar_vectores_por_angulo,
                         guardar_vector_unico, reconocer_persona,
                         eliminar_persona, verificar_duplicado_facial,
                         validar_numero_cuenta, validar_correo,
                         validar_contrasena, validar_grado,
                         validar_grupo, ROLES_VALIDOS)

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

BG      = "#0D0F14"
PANEL   = "#13161E"
CARD    = "#1A1E2A"
ACCENT  = "#00D4FF"
ACCENT2 = "#0099BB"
SUCCESS = "#00FF88"
DANGER  = "#FF3B5C"
WARNING = "#FFB830"
TEXT    = "#E8EAF0"
SUBTEXT = "#6B7280"
BORDER  = "#252A38"

# Paleta inspirada en vista_login (modo oscuro)
NAVY_LN = "#1E4D8C"   # azul marino institucional
TEAL_LN = "#0E9975"   # verde azulado
GOLD_LN = "#B5860D"   # dorado institucional

COLOR_ROL = {"estudiante": ACCENT, "maestro": WARNING, "admin": DANGER}

W, H    = 1024, 600
PANEL_W = 320
CAM_W   = W - PANEL_W

HAAR_PATH = None

GRADOS = [str(i) for i in range(1, 21)]
GRUPOS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

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
                   font=("Courier New", font_size, "bold"),
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
        self.configure(bg=BG)

        self.f_title  = tkfont.Font(family="Courier New", size=16, weight="bold")
        self.f_sub    = tkfont.Font(family="Courier New", size=8)
        self.f_btn    = tkfont.Font(family="Courier New", size=10, weight="bold")
        self.f_label  = tkfont.Font(family="Courier New", size=8)
        self.f_status = tkfont.Font(family="Courier New", size=8,  weight="bold")
        self.f_zona   = tkfont.Font(family="Courier New", size=7)

        self.picam2      = None
        self._cap        = None
        self.cam_running = False
        self.verificando = False

        self._frame_actual = None
        self._frame_lock   = threading.Lock()

        self._analisis = {"vector": None, "coords": None,
                          "frame_id": -1, "tipo": None}
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

        self._build_main()

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
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
                coords = self._analisis["coords"]
                vector = self._analisis["vector"]
                tipo   = self._analisis["tipo"]

            with self._ov_lock:
                ov_color = self._ov_color
                ov_texto = self._ov_texto

            vis = frame.copy()
            if coords:
                if self._modo_acceso:
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
        ultimo_id = -1
        while self.cam_running:
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

            with self._analisis_lock:
                self._analisis["vector"]   = vector
                self._analisis["coords"]   = coords
                self._analisis["frame_id"] = frame_id
                self._analisis["tipo"]     = tipo

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

        # ── Barra superior ────────────────────────────────────────────────────
        cv.create_rectangle(0, 0, W, 54, fill=PANEL, outline="")
        cv.create_rectangle(0, 54, W, 56, fill=ACCENT, outline="")
        cv.create_text(W // 2, 27,
                       text="SISTEMA DE CONTROL DE ACCESO  ·  FACIAL",
                       font=("Courier New", 11, "bold"), fill=TEXT)
        # ── Barra inferior ────────────────────────────────────────────────────
        cv.create_rectangle(0, H - 30, W, H, fill=PANEL, outline="")
        cv.create_rectangle(0, H - 30, W, H - 29, fill=BORDER, outline="")
        cv.create_text(W // 2, H - 15,
                       text=f"Universidad de Colima  ·  Facultad de Ingeniería Electromecanica  "
                            f"|  v5.5  |  {int(TIEMPO_ESCANEO)}s  |  max {MAX_MUESTRAS_PASO} muestras/paso",
                       font=("Courier New", 7), fill=SUBTEXT)

        # ── Subtítulo central ─────────────────────────────────────────────────
        cv.create_text(W // 2, 86,
                       text="Bienvenido — selecciona una opción",
                       font=("Courier New", 10), fill=SUBTEXT)

        # ── Botones horizontales tipo pill (boceto) ───────────────────────────
        BW, BH = 680, 122
        BX = (W - BW) // 2
        BY1 = H // 2 - BH - 26   # centrados verticalmente
        BY2 = H // 2 + 26

        self._horiz_card_btn(cv, BX, BY1, BW, BH,
                             "REGISTRAR",
                             "Nuevo usuario",
                             "Captura biométrica guiada en 4 ángulos",
                             NAVY_LN, self._show_registro)
        self._horiz_card_btn(cv, BX, BY2, BW, BH,
                             "ACCESO",
                             "Verificar identidad",
                             "Reconocimiento facial en tiempo real",
                             TEAL_LN, self._show_acceso)

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
                 font=("Courier New", 13, "bold"),
                 fg=TEXT, bg=CARD
                 ).place(relx=.5, y=82, anchor="center")
        tk.Label(fr, text=subtitulo,
                 font=("Courier New", 8, "bold"),
                 fg=color, bg=CARD
                 ).place(relx=.5, y=102, anchor="center")
        tk.Label(fr, text=desc,
                 font=("Courier New", 8),
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

        # ── Sombra en el canvas de fondo ─────────────────────────────────────
        _round_rect(cv, x+5, y+5, x+w+5, y+h+5,
                    r=r, fill="#060810", outline="#060810", width=0)

        # ── Canvas de la tarjeta (reemplaza Frame + Canvas separados) ─────────
        cc = tk.Canvas(self, width=w, height=h, bg=BG,
                       highlightthickness=0)
        cc.place(x=x, y=y)

        # Cuerpo tarjeta redondeado completo
        _round_rect(cc, 0, 0, w, h,
                    r=r, fill=CARD, outline=BORDER, width=1)

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
        cc.create_rectangle(0, 0, r, r, fill=CARD, outline="")
        cc.create_rectangle(0, h-r, r, h, fill=CARD, outline="")
        # Redibuja arcos del cuerpo sobre la esquina para limpiar
        cc.create_arc(0, 0,    2*r, 2*r,   start=90,  extent=90,
                      fill=color, outline=color)
        cc.create_arc(0, h-2*r, 2*r, h,    start=180, extent=90,
                      fill=color, outline=color)
        cc.create_rectangle(0, r,  r, h-r, fill=color, outline="")

        # ── Ícono biométrico centrado en la banda ─────────────────────────────
        ic_cx, ic_cy = bw // 2, h // 2
        cc.create_oval(ic_cx-30, ic_cy-30, ic_cx+30, ic_cy+30,
                       outline="#FFFFFF", width=2)
        cc.create_oval(ic_cx-13, ic_cy-13, ic_cx+13, ic_cy+13,
                       fill="#FFFFFF", outline="")
        for dx, dy in [(-22, -22), (22, -22), (-22, 22), (22, 22)]:
            cc.create_oval(ic_cx+dx-4, ic_cy+dy-4,
                           ic_cx+dx+4, ic_cy+dy+4,
                           fill="#FFFFFF", outline="")

        # ── Línea separadora vertical ─────────────────────────────────────────
        cc.create_line(bw+10, 14, bw+10, h-14, fill=BORDER, width=1)

        # ── Textos ────────────────────────────────────────────────────────────
        tx = bw + 26
        cc.create_text(tx, 28,  text=titulo,    anchor="w",
                       font=("Courier New", 16, "bold"), fill=TEXT)
        cc.create_text(tx, 56,  text=subtitulo, anchor="w",
                       font=("Courier New", 9, "bold"),  fill=color)
        cc.create_text(tx, 76,  text=desc,      anchor="w",
                       font=("Courier New", 8),          fill=SUBTEXT)

        # ── Botón pill ENTRAR (Canvas embebido via create_window) ─────────────
        btn_w2, btn_h2 = 150, 46
        btn_f = _rounded_btn(cc, "ENTRAR  ▶", cmd,
                             width=btn_w2, height=btn_h2,
                             bg=color, fg="#FFFFFF",
                             hover=hover_c,
                             font_size=9, bg_parent=CARD)
        cc.create_window(w - btn_w2 - 18, (h - btn_h2) // 2,
                         anchor="nw", window=btn_f)

        # ── Card completa clickable ───────────────────────────────────────────
        cc.bind("<Button-1>", lambda e: cmd())

    @staticmethod
    def _lighten_static(hx):
        h = hx.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{min(255,r+40):02x}{min(255,g+40):02x}{min(255,b+40):02x}"

    # ══════════════════════════════════════════════════════════════════════════
    #  PANTALLA REGISTRO
    # ══════════════════════════════════════════════════════════════════════════
    def _show_registro(self):
        self._clear()
        self.geometry(f"{W}x{H}+0+0")
        self._set_overlay(None, "")

        left = tk.Frame(self, bg=PANEL, width=PANEL_W, height=H)
        left.place(x=0, y=0)

        # ── Header del panel ──────────────────────────────────────────────────
        # Franja de acento curva superior (canvas recubre los primeros 58px)
        hdr_cv = tk.Canvas(left, width=PANEL_W, height=58,
                           bg=PANEL, highlightthickness=0)
        hdr_cv.place(x=0, y=0)
        hdr_cv.create_rectangle(0, 0, PANEL_W, 4, fill=ACCENT, outline="")
        hdr_cv.create_text(18, 22,
                           text="◈  REGISTRO",
                           font=("Courier New", 13, "bold"), fill=ACCENT,
                           anchor="w")
        hdr_cv.create_text(18, 42,
                           text="Nuevo usuario · captura biométrica guiada",
                           font=("Courier New", 7), fill=SUBTEXT, anchor="w")
        hdr_cv.create_rectangle(18, 56, PANEL_W - 18, 57,
                                fill=BORDER, outline="")

        # Separador vertical derecho
        tk.Frame(self, bg=BORDER, width=1, height=H).place(x=PANEL_W - 1, y=0)

        # ── Campos fijos para todos (con más espacio entre ellos) ──
        # Aumenté la separación vertical de 36 a 42 píxeles
        self._field(left, "Nombre(s)",                   self.nombre_var,  56)
        self._field(left, "Apellido paterno",             self.ap_pat_var,  96)  # +40
        self._field(left, "Apellido materno",             self.ap_mat_var, 136)  # +40
        self._field(left, "Numero de cuenta (8 digitos)", self.cuenta_var, 176)  # +40

        # ── Rol (más espacio) ──
        # Movido de 204 a 214
        tk.Label(left, text="Rol", font=self.f_label,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=210)
        rf = tk.Frame(left, bg=PANEL)
        rf.place(x=18, y=222, width=284)  # +12 desde el label
        for i, rol in enumerate(ROLES_VALIDOS):
            c = COLOR_ROL.get(rol, ACCENT)
            tk.Radiobutton(rf, text=rol.capitalize(),
                           variable=self.rol_var, value=rol,
                           font=self.f_label, fg=c, bg=PANEL,
                           selectcolor=CARD, activebackground=PANEL,
                           activeforeground=c, cursor="hand2",
                           command=self._actualizar_campos_rol
                           ).grid(row=0, column=i, padx=8)

        # ── Zona condicional (más espacio vertical) ──
        # Antes: y=238, height=50. Ahora: y=250, height=60
        self._frame_cond = tk.Frame(left, bg=PANEL, width=284, height=60)
        self._frame_cond.place(x=18, y=250)
        self._frame_cond.pack_propagate(False)

        # Widgets para admin/maestro
        self._lbl_correo = tk.Label(self._frame_cond, text="Correo",
                                    font=self.f_label, fg=SUBTEXT, bg=PANEL)
        self._ent_correo = tk.Entry(self._frame_cond, textvariable=self.correo_var,
                                    font=self.f_sub, fg=TEXT, bg=CARD,
                                    insertbackground=ACCENT, relief="flat",
                                    highlightthickness=1,
                                    highlightcolor=ACCENT,
                                    highlightbackground=BORDER)
        self._lbl_pwd = tk.Label(self._frame_cond, text="Contrasena",
                                  font=self.f_label, fg=SUBTEXT, bg=PANEL)
        self._ent_pwd = tk.Entry(self._frame_cond, textvariable=self.pwd_var,
                                  show="●", font=self.f_sub, fg=TEXT, bg=CARD,
                                  insertbackground=ACCENT, relief="flat",
                                  highlightthickness=1,
                                  highlightcolor=ACCENT,
                                  highlightbackground=BORDER)

        # Widgets para estudiante
        self._lbl_grado = tk.Label(self._frame_cond, text="Grado",
                                    font=self.f_label, fg=SUBTEXT, bg=PANEL)
        self._om_grado  = tk.OptionMenu(self._frame_cond, self.grado_var, *GRADOS)
        self._om_grado.config(font=self.f_sub, fg=TEXT, bg=CARD,
                               activebackground=CARD, activeforeground=ACCENT,
                               highlightthickness=0, relief="flat")
        self._om_grado["menu"].config(bg=CARD, fg=TEXT, font=self.f_sub)

        self._lbl_grupo = tk.Label(self._frame_cond, text="Grupo",
                                    font=self.f_label, fg=SUBTEXT, bg=PANEL)
        self._om_grupo  = tk.OptionMenu(self._frame_cond, self.grupo_var, *GRUPOS)
        self._om_grupo.config(font=self.f_sub, fg=TEXT, bg=CARD,
                               activebackground=CARD, activeforeground=ACCENT,
                               highlightthickness=0, relief="flat")
        self._om_grupo["menu"].config(bg=CARD, fg=TEXT, font=self.f_sub)

        # Mostrar segun rol inicial
        self._actualizar_campos_rol()

        # ── Separador con etiqueta ────────────────────────────────────────────
        sep_cv = tk.Canvas(left, width=284, height=20,
                           bg=PANEL, highlightthickness=0)
        sep_cv.place(x=18, y=312)
        sep_cv.create_line(0, 10, 80, 10,  fill=BORDER, width=1)
        sep_cv.create_text(92, 10, text="PASOS", font=("Courier New", 7),
                           fill=SUBTEXT, anchor="center")
        sep_cv.create_line(104, 10, 284, 10, fill=BORDER, width=1)

        # ── Indicadores de los 4 pasos (tarjetas redondeadas con Canvas) ──────
        self._paso_frames = []
        paso_w = 56
        paso_gap = 6
        for i, (_, icono, _, etiq, _, _, _, _) in enumerate(PASOS_REGISTRO):
            fx = 18 + i * (paso_w + paso_gap)
            # Canvas que dibuja la tarjeta redondeada
            pf_cv = tk.Canvas(left, width=paso_w, height=52,
                               bg=PANEL, highlightthickness=0)
            pf_cv.place(x=fx, y=334)
            _round_rect(pf_cv, 0, 0, paso_w, 52,
                        r=8, fill=BORDER, outline=BORDER, width=0)

            # Usamos un Frame transparente para los Labels encima del Canvas
            pf = tk.Frame(left, bg=BORDER, width=paso_w, height=52)
            pf.place(x=fx, y=334)
            li = tk.Label(pf, text=str(i + 1), font=self.f_btn,
                          fg=SUBTEXT, bg=BORDER)
            li.place(relx=.5, y=12, anchor="center")
            ln = tk.Label(pf, text=icono, font=self.f_label,
                          fg=SUBTEXT, bg=BORDER)
            ln.place(relx=.5, y=28, anchor="center")
            bpf = tk.Frame(pf, bg="#0D0F14", width=paso_w - 8, height=4)
            bpf.place(x=4, y=44)
            bp = tk.Frame(bpf, bg=SUBTEXT, width=0, height=4)
            bp.place(x=0, y=0)
            self._paso_frames.append((pf, li, ln, bp))

        self.paso_desc_var = tk.StringVar(value="Completa los campos e inicia")
        tk.Label(left, textvariable=self.paso_desc_var,
                 font=self.f_label, fg=ACCENT, bg=PANEL,
                 wraplength=284, justify="center"
                 ).place(x=18, y=394, width=284)

        self.cap_btn = tk.Button(
            left, text="⬤  INICIAR ESCANEO", font=("Courier New", 9, "bold"),
            fg=BG, bg=ACCENT, relief="flat", cursor="hand2",
            padx=8, pady=7, command=self._iniciar_registro,
            bd=0, highlightthickness=0)
        self.cap_btn.place(x=18, y=416, width=284)
        self.cap_btn.bind("<Enter>",
                          lambda e: self.cap_btn.config(bg=self._lighten(ACCENT)))
        self.cap_btn.bind("<Leave>",
                          lambda e: self.cap_btn.config(bg=ACCENT))

        # ── Separador con etiqueta muestras ───────────────────────────────────
        sep2_cv = tk.Canvas(left, width=284, height=20,
                            bg=PANEL, highlightthickness=0)
        sep2_cv.place(x=18, y=452)
        sep2_cv.create_line(0, 10, 66, 10,    fill=BORDER, width=1)
        sep2_cv.create_text(80, 10, text="MUESTRAS", font=("Courier New", 7),
                            fill=SUBTEXT, anchor="center")
        sep2_cv.create_line(94, 10, 284, 10,  fill=BORDER, width=1)

        # ── Barras de muestras ────────────────────────────────────────────────
        self._barra_pasos = []
        barra_w = 56
        for i, etiq in enumerate(["Frente", "Izquierda", "Derecha", "Frente"]):
            fx = 18 + i * (barra_w + 6)
            tk.Label(left, text=etiq, font=self.f_zona,
                     fg=SUBTEXT, bg=PANEL).place(x=fx, y=474)
            bg_b = tk.Frame(left, bg="#0D0F14", width=barra_w, height=6)
            bg_b.place(x=fx, y=486)
            bar = tk.Frame(bg_b, bg=BORDER, width=0, height=6)
            bar.place(x=0, y=0)
            lbl = tk.Label(left, text="0", font=self.f_zona,
                           fg=SUBTEXT, bg=PANEL)
            lbl.place(x=fx + barra_w // 2, y=495, anchor="center")
            self._barra_pasos.append((bar, lbl, barra_w))

        # ── Separador + progreso ──────────────────────────────────────────────
        tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=506)

        self.progreso_var = tk.StringVar(value="")
        self.prog_label   = tk.Label(left, textvariable=self.progreso_var,
                                     font=self.f_status, fg=WARNING,
                                     bg=PANEL, justify="center", wraplength=284)
        self.prog_label.place(x=18, y=512, width=284)

        self.timer_var    = tk.StringVar(value="")
        self.paso_txt_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.timer_var,
                 font=self.f_title, fg=ACCENT, bg=PANEL
                 ).place(x=18, y=548, anchor="nw")
        tk.Label(left, textvariable=self.paso_txt_var,
                 font=self.f_zona, fg=SUBTEXT, bg=PANEL,
                 wraplength=160).place(x=62, y=556, anchor="nw")

        volver_f = _rounded_btn(left, text="◀  Volver", cmd=self._volver,
                                width=110, height=30, bg=BORDER, fg=SUBTEXT,
                                hover=CARD, font_size=8, bg_parent=PANEL)
        volver_f.place(x=18, y=572)

        right = tk.Frame(self, bg=BG, width=CAM_W, height=H)
        right.place(x=PANEL_W, y=0)
        # Marco redondeado visual alrededor de la cámara
        cam_cv = tk.Canvas(right, width=CAM_W, height=H,
                           bg=BG, highlightthickness=0)
        cam_cv.place(x=0, y=0)
        _round_rect(cam_cv, 8, 8, CAM_W - 8, H - 42,
                    r=12, fill="#080A0F", outline=BORDER, width=1)

        self.cam_label = tk.Label(right, bg="#080A0F")
        self.cam_label.place(x=16, y=16, width=CAM_W - 32, height=H - 66)

        tk.Label(right, textvariable=self.status_var,
                 font=self.f_status, fg=ACCENT, bg=BG).place(x=16, y=H - 38)

        self.prog_frame = tk.Frame(right, bg=BORDER, width=CAM_W - 32, height=6)
        self.prog_frame.place(x=16, y=H - 20)
        self.prog_bar = tk.Frame(self.prog_frame, bg=ACCENT, width=0, height=6)
        self.prog_bar.place(x=0, y=0)

        self._start_cam()
        self.cam_running = True
        threading.Thread(target=self._loop_camara,
                         kwargs={"max_w": CAM_W - 32, "max_h": H - 66},
                         daemon=True).start()
        threading.Thread(target=self._loop_analisis, daemon=True).start()

    def _actualizar_campos_rol(self):
        """
        Muestra correo+contrasena para admin/maestro,
        o grado+grupo para estudiante.
        El frame contenedor ahora tiene 60px de altura para mejor espaciado.
        """
        for w in (self._lbl_correo, self._ent_correo,
                  self._lbl_pwd,    self._ent_pwd,
                  self._lbl_grado,  self._om_grado,
                  self._lbl_grupo,  self._om_grupo):
            try: w.place_forget()
            except: pass

        rol = self.rol_var.get()
        if rol in ("admin", "maestro"):
            # Más espacio vertical dentro del frame condicional
            self._lbl_correo.place(x=0,   y=5)   # Centrado verticalmente
            self._ent_correo.place(x=0,   y=22, width=136, height=24)
            self._lbl_pwd.place(  x=148, y=5)
            self._ent_pwd.place(  x=148, y=22, width=136, height=24)
        else:  # estudiante
            self._lbl_grado.place(x=0,   y=5)
            self._om_grado.place( x=0,   y=22, width=70, height=24)
            self._lbl_grupo.place(x=90,  y=5)
            self._om_grupo.place( x=90,  y=22, width=70, height=24)

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
        if not hasattr(self, "_paso_frames") or not self.cam_running:
            return
        paso_w = 56
        try:
            for i, (pf, li, ln, bp) in enumerate(self._paso_frames):
                if i < paso_idx:
                    pf.config(bg=SUCCESS); li.config(fg=BG, bg=SUCCESS)
                    ln.config(fg=BG, bg=SUCCESS); bp.master.config(bg=SUCCESS)
                    bp.config(width=paso_w-4, bg=SUCCESS)
                elif i == paso_idx:
                    pf.config(bg=CARD); li.config(fg=ACCENT, bg=CARD)
                    ln.config(fg=ACCENT, bg=CARD); bp.master.config(bg=BORDER)
                    bp.config(width=int(progreso*(paso_w-4)), bg=ACCENT)
                else:
                    pf.config(bg=BORDER); li.config(fg=SUBTEXT, bg=BORDER)
                    ln.config(fg=SUBTEXT, bg=BORDER); bp.master.config(bg="#111")
                    bp.config(width=0, bg=SUBTEXT)
        except:
            pass

    def _resetear_pasos_ui(self):
        if not hasattr(self, "_paso_frames"):
            return
        try:
            for pf, li, ln, bp in self._paso_frames:
                pf.config(bg=BORDER); li.config(fg=SUBTEXT, bg=BORDER)
                ln.config(fg=SUBTEXT, bg=BORDER); bp.master.config(bg="#111")
                bp.config(width=0)
        except:
            pass

    def _update_barra_paso(self, paso_idx, n_muestras):
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
        self._set_overlay((255, 59, 92), "Ya registrado")
        try:
            self.progreso_var.set(
                f"Rostro ya registrado como:\n{duplicado['nombre']}")
            self.prog_label.config(fg=DANGER)
            self.status_var.set("Registro cancelado.")
            self.cap_btn.config(state="normal", bg=ACCENT,
                                text="INICIAR ESCANEO")
            self.timer_var.set("")
            self.paso_txt_var.set("")
        except:
            pass
        self.after(100, self._resetear_pasos_ui)

    # ── Validacion e inicio ───────────────────────────────────────────────────
    def _iniciar_registro(self):
        nombre = self.nombre_var.get().strip()
        ap_pat = self.ap_pat_var.get().strip()
        cuenta = self.cuenta_var.get().strip()
        correo = self.correo_var.get().strip()
        pwd    = self.pwd_var.get()
        rol    = self.rol_var.get()
        grado  = self.grado_var.get()
        grupo  = self.grupo_var.get()

        def err(msg):
            self.progreso_var.set(msg)
            self.prog_label.config(fg=DANGER)

        if not nombre or not ap_pat:
            return err("Nombre y apellido paterno son obligatorios.")

        ok, msg = validar_numero_cuenta(cuenta)
        if not ok: return err(msg)

        if rol in ("admin", "maestro"):
            ok, msg = validar_correo(correo)
            if not ok: return err(msg)
            ok, msg = validar_contrasena(pwd)
            if not ok: return err(msg)
        else:
            ok, msg = validar_grado(grado)
            if not ok: return err(msg)
            ok, msg = validar_grupo(grupo)
            if not ok: return err(msg)

        self.cap_btn.config(state="disabled", bg=BORDER, text="Escaneando...")
        self.progreso_var.set("Preparando...")
        self.prog_label.config(fg=WARNING)
        self.timer_var.set("")
        self.after(0, self._resetear_pasos_ui)

        threading.Thread(
            target=self._capturar_registro,
            args=(nombre, self.ap_pat_var.get().strip(),
                  self.ap_mat_var.get().strip(),
                  cuenta, correo, pwd, rol, grado, grupo),
            daemon=True).start()

    def _capturar_registro(self, nombre, ap_pat, ap_mat,
                            cuenta, correo, pwd, rol, grado, grupo):
        BAR_W = CAM_W - 16

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
            self.after(0, lambda: self._safe(
                lambda: self.progreso_var.set(
                    "Error al registrar. Verifica los datos.")))
            self.after(0, lambda: self._safe(
                lambda: self.prog_label.config(fg=DANGER)))
            self.after(0, lambda: self._safe(
                lambda: self.cap_btn.config(
                    state="normal", bg=ACCENT, text="INICIAR ESCANEO")))
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
                    frame_id = self._analisis["frame_id"]
                    v        = self._analisis["vector"]
                    tipo_det = self._analisis["tipo"]

                angulo_ok     = (tipo_det == tipo_esperado) and (v is not None)
                cara_presente = (v is not None)
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
                lambda: self.cap_btn.config(
                    state="normal", bg=SUCCESS,
                    text="REGISTRO COMPLETO")))
            for var in (self.nombre_var, self.ap_pat_var, self.ap_mat_var,
                        self.cuenta_var, self.correo_var, self.pwd_var):
                self.after(0, lambda v=var: self._safe(lambda: v.set("")))
            self.after(0, lambda: self._safe(
                lambda: self.rol_var.set("estudiante")))
            self.after(0, lambda: self._safe(
                lambda: self.grado_var.set("1")))
            self.after(0, lambda: self._safe(
                lambda: self.grupo_var.set("A")))
            self.after(3500, lambda: self._safe(
                lambda: self.cap_btn.config(
                    bg=ACCENT, text="INICIAR ESCANEO")))
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
                lambda: self.cap_btn.config(
                    state="normal", bg=ACCENT, text="INICIAR ESCANEO")))
            self.after(0, self._resetear_pasos_ui)

    # ══════════════════════════════════════════════════════════════════════════
    #  PANTALLA ACCESO
    # ══════════════════════════════════════════════════════════════════════════
    def _show_acceso(self):
        self._clear()
        self.geometry(f"{W}x{H}+0+0")
        self.verificando = False
        self._modo_acceso = True
        self._set_overlay(None, "")

        left = tk.Frame(self, bg=PANEL, width=PANEL_W, height=H)
        left.place(x=0, y=0)

        # ── Header del panel ──────────────────────────────────────────────────
        hdr_cv = tk.Canvas(left, width=PANEL_W, height=58,
                           bg=PANEL, highlightthickness=0)
        hdr_cv.place(x=0, y=0)
        hdr_cv.create_rectangle(0, 0, PANEL_W, 4, fill=SUCCESS, outline="")
        hdr_cv.create_text(18, 22,
                           text="◈  ACCESO",
                           font=("Courier New", 13, "bold"), fill=SUCCESS,
                           anchor="w")
        hdr_cv.create_text(18, 42,
                           text="Verificación de identidad en tiempo real",
                           font=("Courier New", 7), fill=SUBTEXT, anchor="w")
        hdr_cv.create_rectangle(18, 56, PANEL_W - 18, 57,
                                fill=BORDER, outline="")

        # Separador vertical derecho
        tk.Frame(self, bg=BORDER, width=1, height=H).place(x=PANEL_W - 1, y=0)

        tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=62)

        self.posicion_var   = tk.StringVar(value="Mira directo a la camara.")
        self.posicion_label = tk.Label(left, textvariable=self.posicion_var,
                 font=self.f_btn, fg=WARNING, bg=PANEL,
                 wraplength=284, justify="center")
        self.posicion_label.place(x=18, y=76, width=284)

        # Separador con etiqueta "RESULTADO"
        sep1 = tk.Canvas(left, width=284, height=20, bg=PANEL, highlightthickness=0)
        sep1.place(x=18, y=224)
        sep1.create_line(0, 10, 60, 10,    fill=BORDER, width=1)
        sep1.create_text(74, 10, text="RESULTADO", font=("Courier New", 7),
                         fill=SUBTEXT, anchor="center")
        sep1.create_line(90, 10, 284, 10,  fill=BORDER, width=1)

        self.resultado_var   = tk.StringVar(value="Esperando...")
        self.resultado_label = tk.Label(
            left, textvariable=self.resultado_var,
            font=("Courier New", 11, "bold"), fg=ACCENT, bg=PANEL,
            wraplength=284, justify="center")
        self.resultado_label.place(x=18, y=248, width=284)

        self.candidato_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.candidato_var,
                 font=self.f_label, fg=TEXT, bg=PANEL,
                 wraplength=284, justify="center"
                 ).place(x=18, y=286, width=284)

        # Separador con etiqueta "SIMILITUD"
        sep2 = tk.Canvas(left, width=284, height=20, bg=PANEL, highlightthickness=0)
        sep2.place(x=18, y=396)
        sep2.create_line(0, 10, 62, 10,    fill=BORDER, width=1)
        sep2.create_text(76, 10, text="SIMILITUD", font=("Courier New", 7),
                         fill=SUBTEXT, anchor="center")
        sep2.create_line(92, 10, 284, 10,  fill=BORDER, width=1)

        # Barra de similitud con fondo oscuro
        self.sim_bg  = tk.Frame(left, bg="#0D0F14", width=284, height=20)
        self.sim_bg.place(x=18, y=420)
        self.sim_bar = tk.Frame(self.sim_bg, bg=BORDER, width=0, height=20)
        self.sim_bar.place(x=0, y=0)
        self.sim_lbl = tk.Label(self.sim_bg, text="", font=self.f_zona,
                                fg=BG, bg=BORDER)
        self.sim_lbl.place(x=4, y=3)

        self.detalle_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.detalle_var, font=self.f_zona,
                 fg=SUBTEXT, bg=PANEL, wraplength=284, justify="center"
                 ).place(x=18, y=448, width=284)

        volver_f = _rounded_btn(left, text="◀  Volver", cmd=self._volver,
                                width=110, height=30, bg=BORDER, fg=SUBTEXT,
                                hover=CARD, font_size=8, bg_parent=PANEL)
        volver_f.place(x=18, y=558)

        right = tk.Frame(self, bg=BG, width=CAM_W, height=H)
        right.place(x=PANEL_W, y=0)
        # Marco redondeado visual alrededor de la cámara
        cam_cv = tk.Canvas(right, width=CAM_W, height=H,
                           bg=BG, highlightthickness=0)
        cam_cv.place(x=0, y=0)
        _round_rect(cam_cv, 8, 8, CAM_W - 8, H - 8,
                    r=12, fill="#080A0F", outline=BORDER, width=1)

        self.cam_label = tk.Label(right, bg="#080A0F")
        self.cam_label.place(x=16, y=16, width=CAM_W - 32, height=H - 32)

        self._start_cam()
        self.cam_running = True
        self._ultima_cara_t = time.time()
        threading.Thread(target=self._loop_camara,
                         kwargs={"max_w": CAM_W - 32, "max_h": H - 32},
                         daemon=True).start()
        threading.Thread(target=self._loop_analisis, daemon=True).start()
        threading.Thread(target=self._monitor_cara,   daemon=True).start()
        threading.Thread(target=self._guia_posicion,  daemon=True).start()
        self.after(1800, self._lanzar_verificacion)

    def _set_sim_bar(self, pct, color):
        w = max(0, min(284, int(pct / 100 * 284)))
        self.sim_bar.config(width=w, bg=color)
        self.sim_lbl.config(text=f" {pct}%", bg=color,
                            fg=BG if w > 30 else color)

    def _lanzar_verificacion(self):
        if not self.cam_running: return
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
        while len(vectores) < 8 and intentos < 120 and self.cam_running:
            intentos += 1
            with self._analisis_lock:
                frame_id = self._analisis["frame_id"]
                v        = self._analisis["vector"]
                tipo     = self._analisis["tipo"]
            if frame_id == ultimo:
                time.sleep(0.04); continue
            ultimo = frame_id
            if v is not None:
                hay_cara = True
                if tipo == TIPO_FRONTAL:
                    vectores.append(v)
            time.sleep(0.04)

        self._set_overlay(None, "")
        self.verificando = False
        if not self.cam_running: return

        if not vectores:
            self.after(0, lambda: self.resultado_var.set(
                "Gira al frente" if hay_cara else "Sin rostro"))
            self.after(0, lambda: self.resultado_label.config(fg=WARNING))
            self.after(0, lambda: self.candidato_var.set(
                "Mira directo a la camara." if hay_cara
                else "Ponte frente a la camara."))
            self.after(0, lambda: self._set_sim_bar(0, BORDER))
            self.after(3000, self._lanzar_verificacion); return

        v_final   = np.mean(vectores, axis=0).astype(np.float32)
        resultado = reconocer_persona(v_final, angulo_nuevo="frontal")

        if resultado is None:
            from database import cargar_vectores_por_angulo as _cvpa
            hay = len(_cvpa()) > 0
            self.after(0, lambda: self.resultado_var.set(
                "ACCESO DENEGADO" if hay else "Sin registros"))
            self.after(0, lambda: self.resultado_label.config(
                fg=DANGER if hay else WARNING))
            self.after(0, lambda: self.candidato_var.set(
                "Persona no reconocida." if hay
                else "No hay usuarios registrados."))
            self.after(0, lambda: self._set_sim_bar(0, BORDER))
            if hay:
                self._set_overlay((255, 59, 92), "Desconocido")
                self.after(0, lambda: servo.denegar())   # ← línea nueva
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
        self.candidato_var.set(
            f"{r['nombre']}\n"
            f"Cuenta: {r.get('numero_cuenta', '---')}\n"
            f"Rol: {r.get('rol', '---').upper()}\n"
            f"Similitud: {r['similitud_pct']}%")
        self.after(0, lambda: self._safe(
            lambda: self.posicion_label.config(fg=SUCCESS)))
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
        self.candidato_var.set(
            f"Mas parecido a:\n{r['nombre']}\n"
            f"Similitud: {r['similitud_pct']}%")
        servo.denegar()
        self.after(4000, self._lanzar_verificacion)

    def _guia_posicion(self):
        """Actualiza la instruccion de posicion en tiempo real."""
        while self.cam_running:
            with self._analisis_lock:
                v    = self._analisis["vector"]
                tipo = self._analisis["tipo"]
            try:
                if v is None:
                    self.posicion_var.set("Acercate a la camara")
                    self.posicion_label.config(fg=WARNING)
                elif tipo == TIPO_FRONTAL:
                    self.posicion_var.set("Listo, mira a la camara")
                    self.posicion_label.config(fg=SUCCESS)
                elif tipo == TIPO_PERFIL_D:
                    self.posicion_var.set(
                        "Estas volteado a tu Izquierda\nVoltea hacia enfrente")
                    self.posicion_label.config(fg=WARNING)
                elif tipo == TIPO_PERFIL_I:
                    self.posicion_var.set(
                        "Estas volteado a tu Derecha\nVoltea hacia enfrente")
                    self.posicion_label.config(fg=WARNING)
            except Exception:
                pass
            time.sleep(0.15)

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
        except:
            pass

    def on_close(self):
        self._stop_cam()
        servo.desconectar()
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