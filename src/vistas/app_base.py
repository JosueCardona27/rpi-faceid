"""
Clase base App: constantes, cámara, análisis y utilidades compartidas.
Todas las vistas heredan o importan desde aquí.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
                          ZONAS_POR_TIPO, ZONAS_FRONTAL,
                          TIPO_FRONTAL, TIPO_PERFIL_D, TIPO_PERFIL_I)
from database   import (registrar_usuario, guardar_vectores_por_angulo,
                         guardar_vector_unico, reconocer_persona,
                         eliminar_persona, ROLES_VALIDOS)

# ── Paleta de colores ─────────────────────────────────────────────────────────
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

COLOR_ROL = {"estudiante": ACCENT, "maestro": WARNING, "admin": DANGER}

# ── Dimensiones ───────────────────────────────────────────────────────────────
W, H    = 1024, 600
PANEL_W = 320
CAM_W   = W - PANEL_W

HAAR_PATH = None

# ── Pasos de registro ─────────────────────────────────────────────────────────
PASOS_REGISTRO = [
    (0, "●", "Mira directo a la camara",
     "FRENTE",    6.0, "frontal", TIPO_FRONTAL,  "Mira directo a la camara"),
    (1, "◀", "Gira tu cabeza a la DERECHA",
     "DERECHA",   6.0, "perfil",  TIPO_PERFIL_D, "Gira mas a tu derecha"),
    (2, "▶", "Gira tu cabeza a la IZQUIERDA",
     "IZQUIERDA", 6.0, "perfil",  TIPO_PERFIL_I, "Gira mas a tu izquierda"),
    (3, "●", "Vuelve al frente",
     "FRENTE",    6.0, "frontal", TIPO_FRONTAL,  "Mira directo a la camara"),
]
N_PASOS           = len(PASOS_REGISTRO)
TIEMPO_ESCANEO    = sum(p[4] for p in PASOS_REGISTRO)   # 24 s
MAX_MUESTRAS_PASO = 20
MUESTRAS_MIN_PASO = 5


def _imgtk(frame, max_w, max_h):
    h0, w0 = frame.shape[:2]
    r  = min(max_w / w0, max_h / h0)
    fr = cv2.resize(frame, (int(w0*r), int(h0*r)))
    return ImageTk.PhotoImage(
        image=Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))


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

        self.nombre_var   = tk.StringVar()
        self.ap_pat_var   = tk.StringVar()
        self.ap_mat_var   = tk.StringVar()
        self.rol_var      = tk.StringVar(value="estudiante")
        self.status_var   = tk.StringVar(value="Listo")

        self._ov_color = None
        self._ov_texto = ""
        self._ov_lock  = threading.Lock()

        self._build_main()

    # ── Utilidades generales ──────────────────────────────────────────────────
    @staticmethod
    def _lighten(hx):
        h = hx.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{min(255,r+40):02x}{min(255,g+40):02x}{min(255,b+40):02x}"

    def _clear(self):
        for w in self.winfo_children():
            w.destroy()

    def _volver(self):
        self._stop_cam()
        self._build_main()

    def _set_overlay(self, color, texto=""):
        with self._ov_lock:
            self._ov_color = color
            self._ov_texto = texto

    def _safe(self, fn):
        if self.cam_running:
            try: fn()
            except: pass

    def _field(self, parent, label, var, y):
        tk.Label(parent, text=label, font=self.f_label,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=y)
        tk.Entry(parent, textvariable=var, font=self.f_sub,
                 fg=TEXT, bg=CARD, insertbackground=ACCENT,
                 relief="flat", highlightthickness=1,
                 highlightcolor=ACCENT, highlightbackground=BORDER
                 ).place(x=18, y=y+14, width=284, height=24)

    def _card_btn(self, cv, x, y, w, h, titulo, desc, color, cmd):
        cv.create_rectangle(x+3, y+3, x+w+3, y+h+3, fill="#080A0F", outline="")
        cv.create_rectangle(x,   y,   x+w,   y+h,   fill=CARD, outline=BORDER, width=1)
        cv.create_rectangle(x,   y,   x+w,   y+3,   fill=color, outline="")
        fr = tk.Frame(self, bg=CARD, width=w, height=h)
        fr.place(x=x, y=y)
        tk.Label(fr, text=titulo, font=self.f_btn,   fg=color,  bg=CARD
                 ).place(relx=.5, y=30,  anchor="center")
        tk.Label(fr, text=desc,   font=self.f_label, fg=SUBTEXT, bg=CARD,
                 justify="center").place(relx=.5, y=76,  anchor="center")
        btn = tk.Button(fr, text="INICIAR  ->", font=self.f_status,
                        fg=BG, bg=color, relief="flat", cursor="hand2",
                        command=cmd, padx=14, pady=5)
        btn.place(relx=.5, y=140, anchor="center")
        btn.bind("<Enter>", lambda e,b=btn,c=color: b.config(bg=self._lighten(c)))
        btn.bind("<Leave>", lambda e,b=btn,c=color: b.config(bg=c))

    # ── Cámara ────────────────────────────────────────────────────────────────
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

    # ── Navegación entre vistas ───────────────────────────────────────────────
    def _build_main(self):
        from vistas.vista_main import build_main
        build_main(self)

    def _show_registro(self):
        from vistas.vista_registro import show_registro
        show_registro(self)

    def _show_acceso(self):
        from vistas.vista_acceso import show_acceso
        show_acceso(self)

    # ── Cierre ────────────────────────────────────────────────────────────────
    def on_close(self):
        self._stop_cam()
        self.destroy()