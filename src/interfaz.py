"""
interfaz.py
===========
Interfaz grafica del sistema de acceso facial.
Optimizada para pantalla tactil 7" 1024x600.
- Menu perfectamente centrado
- Escaneo por tiempo (10s multi-angulo)
- Barras de zonas permanentes: no se borran al retirar el rostro
- Deteccion automatica de camara: Picamera2 (RPi) o webcam OpenCV (laptop)
"""

import tkinter as tk
from tkinter import font as tkfont
import cv2
import numpy as np
import threading
import time
from PIL import Image, ImageTk

# ── deteccion automatica de camara ────────────────────────────────────────────
# Intenta Picamera2 (Raspberry Pi). Si falla usa OpenCV (webcam de laptop).
USAR_PICAM = False
try:
    from picamera2 import Picamera2
    _test = Picamera2()
    _test.close()
    del _test
    USAR_PICAM = True
    print("[CAM] Picamera2 detectada — modo Raspberry Pi")
except Exception:
    print("[CAM] Picamera2 no disponible — modo webcam OpenCV")

from face_engine  import (extraer_caracteristicas, dibujar_overlay,
                           N_ZONAS, ZONAS,
                           TIPO_FRONTAL, TIPO_PERFIL_D, TIPO_PERFIL_I, TIPO_ABAJO)
from database     import (registrar_persona, guardar_vector_unico,
                          reconocer_persona)

# ─── PALETA ───────────────────────────────────────────────────────────────────
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

# ─── RESOLUCION ───────────────────────────────────────────────────────────────
W, H    = 1024, 600
PANEL_W = 300
CAM_W   = W - PANEL_W   # 724

# ─── CONFIGURACION ESCANEO ────────────────────────────────────────────────────
# El detector DNN no necesita archivo XML — usa opencv_face_detector.caffemodel
# que se busca automaticamente en ../models/ relativo a este archivo.
HAAR_PATH    = None   # no se usa con DNN, se mantiene por compatibilidad
MUESTRAS_MIN = 3      # muestras minimas por paso para aceptarlo

# ─── Pasos de registro multi-angulo ───────────────────────────────────────────
# Cada paso: (id, icono, instruccion_camara, etiqueta_panel, segundos,
#             modo_deteccion, tipo_esperado, msg_correccion)
PASOS_REGISTRO = [
    (0, "●", "Mira directo a la camara",
     "FRENTE",       5.0, "frontal",
     TIPO_FRONTAL,  "Mira directo a la camara"),

    (1, "◀", "Gira la cabeza a la DERECHA",
     "DERECHA",      5.0, "perfil",
     TIPO_PERFIL_D, "Gira mas a la derecha"),

    (2, "▶", "Gira la cabeza a la IZQUIERDA",
     "IZQUIERDA",    5.0, "perfil",
     TIPO_PERFIL_I, "Gira mas a la izquierda"),

    (3, "▼", "Inclina la cabeza hacia abajo",
     "ABAJO",        5.0, "abajo",
     TIPO_ABAJO,    "Inclina mas la cabeza hacia abajo"),

    (4, "●", "Vuelve al frente — ultimo paso",
     "FRENTE FINAL", 5.0, "frontal",
     TIPO_FRONTAL,  "Mira directo a la camara"),
]
N_PASOS        = len(PASOS_REGISTRO)
TIEMPO_ESCANEO = sum(p[4] for p in PASOS_REGISTRO)   # 25s total

NOMBRES_ZONA = ["Frente", "Ojo izq", "Ojo der",
                "Nariz", "Mejilla izq", "Mejilla der", "Boca/menton"]


# ── helper ────────────────────────────────────────────────────────────────────
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

        # handles de camara (solo uno estara activo segun USAR_PICAM)
        self.picam2  = None   # Picamera2 en RPi
        self._cap    = None   # cv2.VideoCapture en laptop

        self.cam_running = False
        self.verificando = False

        # buffer compartido: _loop_camara escribe, los hilos de analisis leen
        self._frame_actual = None
        self._frame_lock   = threading.Lock()

        # resultado del analisis facial (hilo _loop_analisis lo actualiza)
        self._analisis      = {"vector": None, "pesos": None,
                               "coords": None, "frame_id": -1,
                               "tipo": None}
        self._analisis_lock = threading.Lock()
        self._modo_deteccion = "auto"   # cambia segun el paso activo

        # barras de zonas
        self._zonas_congeladas  = False
        self._zonas_permanentes = np.zeros(N_ZONAS, dtype=np.float32)
        self._zona_bars         = []
        self._zona_cap_bars     = []
        self._zona_cap_acum     = np.zeros(N_ZONAS, dtype=np.float32)

        self.nombre_var = tk.StringVar()
        self.cuenta_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Listo")

        self._ov_color = None
        self._ov_texto = ""
        self._ov_lock  = threading.Lock()

        self._zonas_vivas = np.zeros(N_ZONAS, dtype=np.float32)
        self._zonas_lock  = threading.Lock()

        self._build_main()

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _lighten(hx):
        h = hx.lstrip("#")
        r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return f"#{min(255,r+40):02x}{min(255,g+40):02x}{min(255,b+40):02x}"

    def _clear(self):
        for w in self.winfo_children():
            w.destroy()

    def _stop_cam(self):
        self.cam_running = False
        time.sleep(0.25)
        if USAR_PICAM:
            if self.picam2:
                try:
                    self.picam2.close()
                except Exception:
                    pass
                self.picam2 = None
        else:
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def _volver(self):
        self._stop_cam()
        self._build_main()

    def _set_overlay(self, color, texto=""):
        with self._ov_lock:
            self._ov_color = color
            self._ov_texto = texto

    def _start_cam(self):
        """Abre la camara correcta segun el sistema detectado al inicio."""
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
            # calentar la camara con algunos frames descartados
            for _ in range(5):
                self._cap.read()

    def _leer_frame(self):
        """Lee un frame crudo de la camara activa. Retorna None si falla."""
        try:
            if USAR_PICAM:
                return self.picam2.capture_array()
            else:
                ret, raw = self._cap.read()
                return raw if ret else None
        except Exception:
            return None

    # ── loop de camara ────────────────────────────────────────────────────────
    def _loop_camara(self, max_w, max_h):
        """
        Hilo 1 — Solo captura frames y programa la actualizacion en el
        hilo principal via after(). NUNCA toca Tkinter directamente.
        """
        self._cam_max_w = max_w
        self._cam_max_h = max_h
        while self.cam_running:
            raw = self._leer_frame()
            if raw is None:
                time.sleep(0.03)
                continue

            frame = cv2.flip(raw, 1)

            # publicar frame para hilo de analisis
            with self._frame_lock:
                self._frame_actual = frame

            # leer ultimo resultado de analisis (no bloquea)
            with self._analisis_lock:
                coords = self._analisis["coords"]
                pesos  = self._analisis["pesos"]
                vector = self._analisis["vector"]
                tipo   = self._analisis["tipo"]

            with self._ov_lock:
                ov_color = self._ov_color
                ov_texto = self._ov_texto

            # dibujar overlay en copia del frame (cpu, no tkinter)
            vis = frame.copy()
            if coords:
                c = ov_color if ov_color else (
                    (0,212,255) if vector is not None else (80,80,80))
                t = ov_texto if ov_texto else (
                    "Detectado" if vector is not None else "Buscando...")
                vis = dibujar_overlay(vis, coords, c, t, tipo=tipo)

                if pesos is not None and vector is not None:
                    x0, y0, w0, h0 = coords
                    for iz, (r0, r1, c0_, c1_, _) in enumerate(ZONAS):
                        zx = int(x0 + c0_ * w0 / 128)
                        zy = int(y0 + r0  * h0 / 128)
                        zw = int((c1_ - c0_) * w0 / 128)
                        zh = int((r1  - r0)  * h0 / 128)
                        if pesos[iz] > 0.1:
                            a = pesos[iz]
                            cv2.rectangle(vis, (zx,zy),(zx+zw,zy+zh),
                                (int(255*(1-a)), int(200*a), int(255*a)), 1)

            # convertir a ImageTk en el hilo (cpu, no tkinter)
            imgtk = _imgtk(vis, max_w, max_h)

            # UNICA llamada a tkinter: programada en el hilo principal
            self.after(0, self._mostrar_frame, imgtk)

            time.sleep(0.033)   # ~30 fps

    def _mostrar_frame(self, imgtk):
        """Ejecuta en el hilo principal de Tkinter — sin parpadeo."""
        if not self.cam_running:
            return
        try:
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)
        except Exception:
            pass

    # ── loop de analisis facial (hilo separado, no bloquea el display) ────────
    def _loop_analisis(self):
        """
        Hilo 2 — Detector Haar + LBP en segundo plano.
        Usa self._modo_deteccion para saber que angulo buscar.
        """
        ultimo_id = -1
        while self.cam_running:
            with self._frame_lock:
                frame    = self._frame_actual
                frame_id = id(frame) if frame is not None else -1

            if frame is None or frame_id == ultimo_id:
                time.sleep(0.02)
                continue

            ultimo_id = frame_id
            modo = self._modo_deteccion
            resultado = extraer_caracteristicas(frame, HAAR_PATH, modo=modo)
            vector, pesos, coords, tipo = resultado

            with self._analisis_lock:
                self._analisis["vector"]   = vector
                self._analisis["pesos"]    = pesos
                self._analisis["coords"]   = coords
                self._analisis["frame_id"] = frame_id
                self._analisis["tipo"]     = tipo

            if pesos is not None and not self._zonas_congeladas:
                with self._zonas_lock:
                    self._zonas_vivas = pesos.copy()

    # ── tick zonas ────────────────────────────────────────────────────────────
    def _tick_zonas(self):
        if not self.cam_running:
            return
        if hasattr(self, "_zona_bars") and self._zona_bars:
            pesos = (self._zonas_permanentes if self._zonas_congeladas
                     else self._zonas_vivas.copy())
            for iz, (bar, _) in enumerate(self._zona_bars):
                p   = float(pesos[iz]) if iz < len(pesos) else 0.0
                w   = int(p * 80)
                col = (SUCCESS if self._zonas_congeladas and p > 0.3 else
                       SUCCESS if p > 0.7 else
                       WARNING if p > 0.3 else
                       DANGER  if p > 0.1 else BORDER)
                bar.config(width=w, bg=col)
        self.after(120, self._tick_zonas)

    def _fijar_zonas(self, pesos):
        self._zonas_permanentes = np.array(pesos, dtype=np.float32)
        self._zonas_congeladas  = True

    def _liberar_zonas(self):
        self._zonas_congeladas = False

    # ══════════════════════════════════════════════════════════════════════════
    #  PANTALLA PRINCIPAL
    # ══════════════════════════════════════════════════════════════════════════
    def _build_main(self):
        self._clear()
        self.geometry(f"{W}x{H}+0+0")
        self._zonas_congeladas = False

        cv = tk.Canvas(self, width=W, height=H, bg=BG, highlightthickness=0)
        cv.place(x=0, y=0)
        cv.create_rectangle(0, 0, W, 3, fill=ACCENT, outline="")

        IY = 30
        cx = W // 2
        cv.create_oval(cx-20, IY,    cx+20, IY+44, outline=ACCENT,  width=2)
        cv.create_oval(cx-11, IY+9,  cx+11, IY+33, outline=ACCENT2, width=2)
        cv.create_line(cx,    IY+44, cx,    IY+62,  fill=ACCENT, width=2)
        cv.create_line(cx-20, IY+53, cx+20, IY+53,  fill=ACCENT, width=2)

        TY = IY + 74
        tk.Label(self, text="SISTEMA DE ACCESO FACIAL",
                 font=self.f_title, fg=ACCENT, bg=BG
                 ).place(x=W//2, y=TY, anchor="center")

        SY = TY + 22
        modo_txt = "Raspberry Pi (Picamera2)" if USAR_PICAM else "Laptop (Webcam)"
        tk.Label(self,
                 text=f"Reconocimiento LBP zonal  |  Multi-angulo  |  {modo_txt}",
                 font=self.f_sub, fg=SUBTEXT, bg=BG
                 ).place(x=W//2, y=SY, anchor="center")

        LY = SY + 16
        cv.create_line(W//2-280, LY, W//2+280, LY, fill=BORDER, width=1)

        CY = LY + 16
        CW, CH  = 230, 180
        GAP     = 40
        total_w = CW*2 + GAP
        CX1 = W//2 - total_w//2
        CX2 = CX1 + CW + GAP

        self._card_btn(cv, CX1, CY, CW, CH, "REGISTRARME",
                       "Registrar nuevo usuario\ny capturar datos faciales",
                       ACCENT, self._show_registro)
        self._card_btn(cv, CX2, CY, CW, CH, "ACCESO",
                       "Verificar identidad\nmediante reconocimiento facial",
                       SUCCESS, self._show_acceso)

        cv.create_line(0, H-26, W, H-26, fill=BORDER, width=1)
        tk.Label(self,
                 text=f"v5.1  |  LBP x7 zonas  |  Escaneo {int(TIEMPO_ESCANEO)}s  |  {'RPi' if USAR_PICAM else 'Webcam'}",
                 font=self.f_zona, fg=SUBTEXT, bg=BG
                 ).place(x=W//2, y=H-13, anchor="center")

    def _card_btn(self, cv, x, y, w, h, titulo, desc, color, cmd):
        cv.create_rectangle(x+3, y+3, x+w+3, y+h+3, fill="#080A0F", outline="")
        cv.create_rectangle(x,   y,   x+w,   y+h,   fill=CARD, outline=BORDER, width=1)
        cv.create_rectangle(x,   y,   x+w,   y+3,   fill=color, outline="")
        fr = tk.Frame(self, bg=CARD, width=w, height=h)
        fr.place(x=x, y=y)
        tk.Label(fr, text=titulo, font=self.f_btn, fg=color, bg=CARD
                 ).place(relx=.5, y=30, anchor="center")
        tk.Label(fr, text=desc, font=self.f_label, fg=SUBTEXT, bg=CARD,
                 justify="center").place(relx=.5, y=76, anchor="center")
        btn = tk.Button(fr, text="INICIAR  ->", font=self.f_status,
                        fg=BG, bg=color, relief="flat", cursor="hand2",
                        command=cmd, padx=14, pady=5)
        btn.place(relx=.5, y=140, anchor="center")
        btn.bind("<Enter>", lambda e,b=btn,c=color: b.config(bg=self._lighten(c)))
        btn.bind("<Leave>", lambda e,b=btn,c=color: b.config(bg=c))

    # ══════════════════════════════════════════════════════════════════════════
    #  PANTALLA REGISTRO
    # ══════════════════════════════════════════════════════════════════════════
    def _show_registro(self):
        self._clear()
        self.geometry(f"{W}x{H}+0+0")
        self._set_overlay(None, "")
        self._zonas_congeladas = False

        left = tk.Frame(self, bg=PANEL, width=PANEL_W, height=H)
        left.place(x=0, y=0)
        tk.Frame(left, bg=ACCENT, width=PANEL_W, height=3).place(x=0, y=0)

        tk.Label(left, text="REGISTRO", font=self.f_title,
                 fg=ACCENT, bg=PANEL).place(x=18, y=8)
        tk.Label(left, text="Nuevo usuario", font=self.f_sub,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=34)
        tk.Frame(left, bg=BORDER, height=1, width=264).place(x=18, y=52)

        self._field(left, "Nombre completo",  self.nombre_var, 60)
        self._field(left, "Numero de cuenta", self.cuenta_var, 108)

        tk.Frame(left, bg=BORDER, height=1, width=264).place(x=18, y=152)

        # ── indicadores de pasos ──────────────────────────────────────────────
        tk.Label(left, text="PASOS DE ESCANEO:", font=self.f_zona,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=158)

        self._paso_frames = []  # (frame, lbl_num, lbl_etiq, barra_prog)
        paso_w = 46
        for i, (_, icono, _, etiqueta, _, _, _, _) in enumerate(PASOS_REGISTRO):
            fx = 18 + i * (paso_w + 4)
            pf = tk.Frame(left, bg=BORDER, width=paso_w, height=56)
            pf.place(x=fx, y=172)
            # numero del paso
            li = tk.Label(pf, text=str(i+1), font=self.f_btn,
                          fg=SUBTEXT, bg=BORDER)
            li.place(relx=.5, y=10, anchor="center")
            # icono/etiqueta
            ln = tk.Label(pf, text=icono, font=self.f_label,
                          fg=SUBTEXT, bg=BORDER)
            ln.place(relx=.5, y=30, anchor="center")
            # mini barra
            bpf = tk.Frame(pf, bg="#111", width=paso_w-4, height=4)
            bpf.place(x=2, y=50)
            bp = tk.Frame(bpf, bg=SUBTEXT, width=0, height=4)
            bp.place(x=0, y=0)
            self._paso_frames.append((pf, li, ln, bp))

        # descripcion del paso actual (texto largo)
        self.paso_desc_var = tk.StringVar(value="Llena los campos e inicia el escaneo")
        tk.Label(left, textvariable=self.paso_desc_var,
                 font=self.f_label, fg=ACCENT, bg=PANEL,
                 wraplength=264, justify="center"
                 ).place(x=18, y=234, width=264)

        tk.Frame(left, bg=BORDER, height=1, width=264).place(x=18, y=254)

        self.cap_btn = tk.Button(
            left, text="INICIAR ESCANEO", font=self.f_btn,
            fg=BG, bg=ACCENT, relief="flat", cursor="hand2",
            padx=8, pady=7, command=self._iniciar_registro)
        self.cap_btn.place(x=18, y=260, width=264)
        self.cap_btn.bind("<Enter>",
            lambda e: self.cap_btn.config(bg=self._lighten(ACCENT)))
        self.cap_btn.bind("<Leave>",
            lambda e: self.cap_btn.config(bg=ACCENT))

        tk.Frame(left, bg=BORDER, height=1, width=264).place(x=18, y=304)

        # ── doble columna de barras de zonas ──────────────────────────────────
        tk.Label(left, text="EN VIVO",    font=self.f_zona,
                 fg=SUBTEXT, bg=PANEL).place(x=112, y=308)
        tk.Label(left, text="CAPTURADAS", font=self.f_zona,
                 fg=SUCCESS,  bg=PANEL).place(x=186, y=308)

        self._zona_bars     = []
        self._zona_cap_bars = []
        self._zona_cap_acum = np.zeros(N_ZONAS, dtype=np.float32)

        for iz, nombre in enumerate(NOMBRES_ZONA):
            yp = 320 + iz * 16
            tk.Label(left, text=f"{nombre[:9]}", font=self.f_zona,
                     fg=SUBTEXT, bg=PANEL).place(x=18, y=yp)
            bg_vivo = tk.Frame(left, bg=BORDER, width=56, height=8)
            bg_vivo.place(x=110, y=yp+2)
            bar_vivo = tk.Frame(bg_vivo, bg=BORDER, width=0, height=8)
            bar_vivo.place(x=0, y=0)
            self._zona_bars.append((bar_vivo, None))
            bg_cap = tk.Frame(left, bg=BORDER, width=56, height=8)
            bg_cap.place(x=190, y=yp+2)
            bar_cap = tk.Frame(bg_cap, bg=BORDER, width=0, height=8)
            bar_cap.place(x=0, y=0)
            self._zona_cap_bars.append(bar_cap)

        tk.Frame(left, bg=BORDER, height=1, width=264).place(x=18, y=434)

        self.progreso_var = tk.StringVar(value="")
        self.prog_label   = tk.Label(left, textvariable=self.progreso_var,
                                     font=self.f_status, fg=WARNING, bg=PANEL,
                                     justify="center", wraplength=264)
        self.prog_label.place(x=18, y=440, width=264)

        # timer + paso actual
        self.timer_var    = tk.StringVar(value="")
        self.paso_txt_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.timer_var,
                 font=self.f_title, fg=ACCENT, bg=PANEL
                 ).place(x=18, y=480, anchor="nw")
        tk.Label(left, textvariable=self.paso_txt_var,
                 font=self.f_zona, fg=SUBTEXT, bg=PANEL,
                 wraplength=160).place(x=62, y=488, anchor="nw")

        tk.Button(left, text="<  Volver", font=self.f_label,
                  fg=SUBTEXT, bg=PANEL, relief="flat",
                  cursor="hand2", command=self._volver).place(x=18, y=556)

        # ── camara ────────────────────────────────────────────────────────────
        right = tk.Frame(self, bg=BG, width=CAM_W, height=H)
        right.place(x=PANEL_W, y=0)
        self.cam_label = tk.Label(right, bg="#080A0F")
        self.cam_label.place(x=8, y=5, width=CAM_W-16, height=H-46)

        tk.Label(right, textvariable=self.status_var,
                 font=self.f_status, fg=ACCENT, bg=BG).place(x=8, y=H-36)

        self.prog_frame = tk.Frame(right, bg=BORDER, width=CAM_W-16, height=8)
        self.prog_frame.place(x=8, y=H-18)
        self.prog_bar = tk.Frame(self.prog_frame, bg=ACCENT, width=0, height=8)
        self.prog_bar.place(x=0, y=0)

        self._start_cam()
        self.cam_running = True
        threading.Thread(target=self._loop_camara,
                         kwargs={"max_w": CAM_W-16, "max_h": H-46},
                         daemon=True).start()
        threading.Thread(target=self._loop_analisis,
                         daemon=True).start()
        self.after(150, self._tick_zonas)

    def _field(self, parent, label, var, y):
        tk.Label(parent, text=label, font=self.f_label,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=y)
        tk.Entry(parent, textvariable=var, font=self.f_sub,
                 fg=TEXT, bg=CARD, insertbackground=ACCENT,
                 relief="flat", highlightthickness=1,
                 highlightcolor=ACCENT,
                 highlightbackground=BORDER
                 ).place(x=18, y=y+15, width=264, height=26)

    def _reset_cap_bars(self):
        if hasattr(self, "_zona_cap_bars"):
            for bar in self._zona_cap_bars:
                bar.config(width=0, bg=BORDER)

    def _update_cap_bars(self, pesos):
        if hasattr(self, "_zona_cap_bars"):
            for iz, bar in enumerate(self._zona_cap_bars):
                p = float(pesos[iz]) if iz < len(pesos) else 0.0
                w = int(p * 56)
                if w > 0:
                    col = SUCCESS if p > 0.6 else WARNING if p > 0.3 else ACCENT
                    bar.config(width=w, bg=col)

    def _activar_paso_ui(self, paso_idx, progreso_paso=0.0):
        """Actualiza visualmente los indicadores de pasos."""
        if not hasattr(self, "_paso_frames"):
            return
        paso_w = 46
        for i, (pf, li, ln, bp) in enumerate(self._paso_frames):
            if i < paso_idx:
                # completado — verde
                pf.config(bg=SUCCESS)
                li.config(fg=BG, bg=SUCCESS)
                ln.config(fg=BG, bg=SUCCESS)
                bp.master.config(bg=SUCCESS)
                bp.config(width=paso_w-4, bg=SUCCESS)
            elif i == paso_idx:
                # activo — azul con barra que avanza
                pf.config(bg=CARD)
                li.config(fg=ACCENT, bg=CARD)
                ln.config(fg=ACCENT, bg=CARD)
                bp.master.config(bg=BORDER)
                bp.config(width=int(progreso_paso*(paso_w-4)), bg=ACCENT)
            else:
                # pendiente — gris
                pf.config(bg=BORDER)
                li.config(fg=SUBTEXT, bg=BORDER)
                ln.config(fg=SUBTEXT, bg=BORDER)
                bp.master.config(bg="#111")
                bp.config(width=0, bg=SUBTEXT)

    def _resetear_pasos_ui(self):
        if not hasattr(self, "_paso_frames"):
            return
        paso_w = 46
        for pf, li, ln, bp in self._paso_frames:
            pf.config(bg=BORDER)
            li.config(fg=SUBTEXT, bg=BORDER)
            ln.config(fg=SUBTEXT, bg=BORDER)
            bp.master.config(bg="#111")
            bp.config(width=0)

    # ── registro: escaneo multi-paso ──────────────────────────────────────────
    def _iniciar_registro(self):
        nombre = self.nombre_var.get().strip()
        cuenta = self.cuenta_var.get().strip()
        if not nombre or not cuenta:
            self.progreso_var.set("Completa nombre y cuenta.")
            self.prog_label.config(fg=DANGER)
            return
        self.cap_btn.config(state="disabled", bg=BORDER, text="Escaneando...")
        self.progreso_var.set("Preparando escaneo...")
        self.prog_label.config(fg=WARNING)
        self.timer_var.set("")
        self.after(0, self._resetear_pasos_ui)
        threading.Thread(target=self._capturar_registro,
                         args=(nombre, cuenta), daemon=True).start()

    def _capturar_registro(self, nombre, cuenta):
        BAR_W = CAM_W - 16

        pid = registrar_persona(cuenta, nombre)
        if pid == -1:
            self.after(0, lambda: self.progreso_var.set(
                "Numero de cuenta ya existe."))
            self.after(0, lambda: self.prog_label.config(fg=DANGER))
            self.after(0, lambda: self.cap_btn.config(
                state="normal", bg=ACCENT, text="INICIAR ESCANEO"))
            return

        # acumuladores globales (todos los pasos)
        todos_vectores = []
        todos_pesos    = []
        self._zona_cap_acum = np.zeros(N_ZONAS, dtype=np.float32)
        self.after(0, self._reset_cap_bars)

        # calcular offsets de tiempo de cada paso
        t_offsets = []
        acum = 0.0
        for p in PASOS_REGISTRO:
            t_offsets.append(acum)
            acum += p[4]

        t_global_inicio = time.time()
        ultimo_analisis = -1

        for paso_idx, (_, icono, instruccion, etiqueta,
                       duracion, modo_det,
                       tipo_esperado, msg_correccion) in enumerate(PASOS_REGISTRO):
            vectores_paso = []
            t_paso_inicio = time.time()
            t_paso_fin    = t_paso_inicio + duracion
            t_global_offset = t_offsets[paso_idx]

            # activar modo de deteccion para este paso
            self._modo_deteccion = modo_det

            desc_paso = f"PASO {paso_idx+1}/{N_PASOS} — {etiqueta}"
            self.after(0, lambda i=instruccion: self.status_var.set(i))
            self.after(0, lambda d=desc_paso: self.paso_desc_var.set(d)
                       if hasattr(self, "paso_desc_var") else None)
            self.after(0, lambda pi=paso_idx: self._activar_paso_ui(pi, 0.0))
            self._set_overlay((255,184,48),
                              f"PASO {paso_idx+1}/{N_PASOS}: {instruccion}")

            while time.time() < t_paso_fin and self.cam_running:
                elapsed_paso   = time.time() - t_paso_inicio
                elapsed_global = t_global_offset + elapsed_paso
                progreso_paso  = elapsed_paso / duracion
                progreso_total = elapsed_global / TIEMPO_ESCANEO
                restante_paso  = max(0.0, duracion - elapsed_paso)

                seg = int(restante_paso) + 1
                self.after(0, lambda s=seg: self.timer_var.set(f"{s}s"))
                self.after(0, lambda pt=int(progreso_total*BAR_W):
                           self.prog_bar.config(width=pt))
                self.after(0, lambda pi=paso_idx, pp=progreso_paso:
                           self._activar_paso_ui(pi, pp))
                self.after(0, lambda pi=paso_idx, e=etiqueta:
                           self.paso_txt_var.set(f"{pi+1}/{N_PASOS} — {e}")
                           if hasattr(self, "paso_txt_var") else None)

                # leer resultado del buffer de analisis
                with self._analisis_lock:
                    frame_id = self._analisis["frame_id"]
                    v        = self._analisis["vector"]
                    p        = self._analisis["pesos"]
                    tipo_det = self._analisis["tipo"]

                if frame_id != ultimo_analisis:
                    ultimo_analisis = frame_id

                    if v is not None and p is not None:
                        # ── validar que el angulo detectado es el correcto ────
                        tipo_ok = (tipo_det == tipo_esperado)

                        if tipo_ok and np.sum(p > 0.15) >= 2:
                            # muestra VALIDA para este paso
                            vectores_paso.append(v)
                            todos_vectores.append(v)
                            todos_pesos.append(p)

                            for iz in range(N_ZONAS):
                                if p[iz] > self._zona_cap_acum[iz]:
                                    self._zona_cap_acum[iz] = p[iz]

                            snap = self._zona_cap_acum.copy()
                            self.after(0, lambda s=snap: self._update_cap_bars(s))

                            n_total = len(todos_vectores)
                            zv = int(np.sum(p > 0.15))
                            self._set_overlay(
                                (0, 255, 136),
                                f"PASO {paso_idx+1}/{N_PASOS} {etiqueta} [{len(vectores_paso)} ok]")
                            self.after(0, lambda nt=n_total, zv=zv:
                                       self.progreso_var.set(
                                           f"Total: {nt} muestras  |  {zv}/7 zonas"))
                            self.after(0, lambda: self.prog_label.config(fg=SUCCESS))

                        else:
                            # angulo incorrecto — guiar al usuario
                            self._set_overlay(
                                (255, 59, 92),
                                f"{msg_correccion}")
                            self.after(0, lambda mc=msg_correccion:
                                       self.status_var.set(mc))
                            self.after(0, lambda: self.prog_label.config(fg=DANGER))

                    else:
                        # no hay cara detectada
                        self._set_overlay(
                            (255, 184, 48),
                            f"PASO {paso_idx+1}/{N_PASOS}: {instruccion}")
                        self.after(0, lambda: self.prog_label.config(fg=WARNING))

                time.sleep(0.04)

            # paso completado — marcarlo verde
            self.after(0, lambda pi=paso_idx: self._activar_paso_ui(pi+1, 0.0))

            # aviso si el paso tuvo pocas muestras
            if len(vectores_paso) < MUESTRAS_MIN:
                self.after(0, lambda e=etiqueta, n=len(vectores_paso):
                           self.progreso_var.set(
                               f"Paso {e}: solo {n} muestras.\n"
                               f"Mantente mas cerca de la camara."))

        # ── fin de todos los pasos ────────────────────────────────────────────
        self._modo_deteccion = "auto"   # restaurar para acceso
        self._set_overlay(None, "")
        self.after(0, lambda: self.timer_var.set(""))
        self.after(0, lambda: self.paso_txt_var.set(""))
        self.after(0, lambda: self.prog_bar.config(width=BAR_W))

        if len(todos_vectores) >= MUESTRAS_MIN * N_PASOS:
            p_final = np.mean(todos_pesos, axis=0).astype(np.float32)
            guardar_vector_unico(pid, todos_vectores, todos_pesos)

            self.after(0, lambda pf=p_final: self._fijar_zonas(pf))
            self.after(0, lambda: self.prog_bar.config(bg=SUCCESS))
            self.after(0, lambda: self.status_var.set(
                f"Registro completo — {len(todos_vectores)} muestras en {N_PASOS} poses"))
            self.after(0, lambda nv=nombre: self.progreso_var.set(
                f"Listo. {nv} registrado\n"
                f"con {len(todos_vectores)} muestras."))
            self.after(0, lambda: self.prog_label.config(fg=SUCCESS))
            self.after(0, lambda: self.cap_btn.config(
                state="normal", bg=SUCCESS, text="REGISTRO COMPLETO ✓"))
            self.after(0, lambda: self.nombre_var.set(""))
            self.after(0, lambda: self.cuenta_var.set(""))
            self.after(3500, lambda: self.cap_btn.config(
                bg=ACCENT, text="INICIAR ESCANEO"))
        else:
            try:
                from database import eliminar_persona
                eliminar_persona(pid)
            except Exception:
                pass
            self.after(0, lambda nv=len(todos_vectores): self.progreso_var.set(
                f"Solo {nv} muestras totales.\nIntentalo de nuevo."))
            self.after(0, lambda: self.prog_label.config(fg=DANGER))
            self.after(0, lambda: self.status_var.set(
                "Acercate mas a la camara e intentalo de nuevo"))
            self.after(0, lambda: self.cap_btn.config(
                state="normal", bg=ACCENT, text="INICIAR ESCANEO"))
            self.after(0, self._resetear_pasos_ui)

    # ══════════════════════════════════════════════════════════════════════════
    #  PANTALLA ACCESO
    # ══════════════════════════════════════════════════════════════════════════
    def _show_acceso(self):
        self._clear()
        self.geometry(f"{W}x{H}+0+0")
        self.verificando = False
        self._set_overlay(None, "")
        self._zonas_congeladas = False

        left = tk.Frame(self, bg=PANEL, width=PANEL_W, height=H)
        left.place(x=0, y=0)
        tk.Frame(left, bg=SUCCESS, width=PANEL_W, height=3).place(x=0, y=0)

        tk.Label(left, text="ACCESO", font=self.f_title,
                 fg=SUCCESS, bg=PANEL).place(x=18, y=8)
        tk.Label(left, text="Verificacion de identidad", font=self.f_sub,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=34)
        tk.Frame(left, bg=BORDER, height=1, width=264).place(x=18, y=52)

        tk.Label(left,
                 text=("La camara escaneara tu\nrostro automaticamente.\n\n"
                       "Funciona con:\n"
                       "  - Lentes\n"
                       "  - Cubrebocas\n"
                       "  - Gorra\n\n"
                       "Solo se usan las\nzonas visibles."),
                 font=self.f_label, fg=SUBTEXT, bg=PANEL,
                 justify="left").place(x=18, y=60)

        tk.Frame(left, bg=BORDER, height=1, width=264).place(x=18, y=208)
        tk.Label(left, text="ZONAS DEL ULTIMO ESCANEO:", font=self.f_zona,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=215)
        self._zona_bars = []
        for iz, nombre in enumerate(NOMBRES_ZONA):
            yp = 230 + iz * 17
            tk.Label(left, text=f"{nombre[:10]:<10}", font=self.f_zona,
                     fg=SUBTEXT, bg=PANEL).place(x=18, y=yp)
            bg_f = tk.Frame(left, bg=BORDER, width=80, height=8)
            bg_f.place(x=112, y=yp+2)
            bar = tk.Frame(bg_f, bg=BORDER, width=0, height=8)
            bar.place(x=0, y=0)
            self._zona_bars.append((bar, None))

        tk.Frame(left, bg=BORDER, height=1, width=264).place(x=18, y=352)

        self.resultado_var   = tk.StringVar(value="Esperando...")
        self.resultado_label = tk.Label(left, textvariable=self.resultado_var,
                                        font=self.f_btn, fg=ACCENT, bg=PANEL,
                                        wraplength=264, justify="center")
        self.resultado_label.place(x=18, y=360, width=264)

        self.candidato_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.candidato_var,
                 font=self.f_label, fg=TEXT, bg=PANEL,
                 wraplength=264, justify="center"
                 ).place(x=18, y=398, width=264)

        self.sim_bg  = tk.Frame(left, bg=BORDER, width=264, height=14)
        self.sim_bg.place(x=18, y=462)
        self.sim_bar = tk.Frame(self.sim_bg, bg=BORDER, width=0, height=14)
        self.sim_bar.place(x=0, y=0)
        self.sim_lbl = tk.Label(self.sim_bg, text="", font=self.f_zona,
                                fg=BG, bg=BORDER)
        self.sim_lbl.place(x=4, y=1)

        self.detalle_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.detalle_var, font=self.f_zona,
                 fg=SUBTEXT, bg=PANEL, wraplength=264,
                 justify="center").place(x=18, y=484, width=264)

        tk.Button(left, text="<  Volver", font=self.f_label,
                  fg=SUBTEXT, bg=PANEL, relief="flat",
                  cursor="hand2", command=self._volver).place(x=18, y=556)

        right = tk.Frame(self, bg=BG, width=CAM_W, height=H)
        right.place(x=PANEL_W, y=0)
        self.cam_label = tk.Label(right, bg="#080A0F")
        self.cam_label.place(x=8, y=5, width=CAM_W-16, height=H-10)

        self._start_cam()
        self.cam_running = True
        threading.Thread(target=self._loop_camara,
                         kwargs={"max_w": CAM_W-16, "max_h": H-10},
                         daemon=True).start()
        threading.Thread(target=self._loop_analisis,
                         daemon=True).start()
        self.after(150,  self._tick_zonas)
        self.after(1800, self._lanzar_verificacion)

    def _set_sim_bar(self, pct, color):
        w = max(0, min(264, int(pct / 100 * 264)))
        self.sim_bar.config(width=w, bg=color)
        self.sim_lbl.config(text=f" {pct}%", bg=color,
                            fg=BG if w > 30 else color)

    # ── ciclo verificacion ────────────────────────────────────────────────────
    def _lanzar_verificacion(self):
        if not self.cam_running:
            return
        if self.verificando:
            self.after(300, self._lanzar_verificacion)
            return
        self.verificando = True
        self._liberar_zonas()
        self.after(0, lambda: self.resultado_var.set("Escaneando..."))
        self.after(0, lambda: self.resultado_label.config(fg=ACCENT))
        self.after(0, lambda: self.candidato_var.set(""))
        self.after(0, lambda: self.detalle_var.set(""))
        self._set_overlay((255,184,48), "Analizando...")
        threading.Thread(target=self._verificar, daemon=True).start()

    def _verificar(self):
        vectores = []
        pesos_l  = []
        intentos = 0
        ultimo   = -1

        while len(vectores) < 8 and intentos < 120 and self.cam_running:
            intentos += 1

            with self._analisis_lock:
                frame_id = self._analisis["frame_id"]
                v        = self._analisis["vector"]
                p        = self._analisis["pesos"]

            if frame_id == ultimo:
                time.sleep(0.04)
                continue

            ultimo = frame_id
            if v is not None and p is not None and np.sum(p > 0.15) >= 2:
                vectores.append(v)
                pesos_l.append(p)
            time.sleep(0.04)

        self._set_overlay(None, "")
        self.verificando = False

        if not self.cam_running:
            return

        if not vectores:
            self.after(0, lambda: self.resultado_var.set("Sin rostro"))
            self.after(0, lambda: self.resultado_label.config(fg=WARNING))
            self.after(0, lambda: self.candidato_var.set(
                "Ponte frente a la camara."))
            self.after(0, lambda: self._set_sim_bar(0, BORDER))
            self.after(3000, self._lanzar_verificacion)
            return

        v_final = np.mean(vectores, axis=0).astype(np.float32)
        p_final = np.mean(pesos_l,  axis=0).astype(np.float32)
        zonas_v = int(np.sum(p_final > 0.15))

        self.after(0, lambda pf=p_final: self._fijar_zonas(pf))

        resultado = reconocer_persona(v_final, p_final)

        if resultado is None:
            self.after(0, lambda: self.resultado_var.set("Sin registros"))
            self.after(0, lambda: self.resultado_label.config(fg=WARNING))
            self.after(0, lambda: self.candidato_var.set(
                "No hay usuarios registrados."))
            self.after(0, lambda: self._set_sim_bar(0, BORDER))
            self.after(4000, self._lanzar_verificacion)
            return

        sim         = resultado["similitud_pct"]
        color_barra = (SUCCESS if resultado["acceso"] else
                       WARNING if sim >= 55  else DANGER)

        self.after(0, lambda: self._set_sim_bar(int(sim), color_barra))
        self.after(0, lambda z=zonas_v: self.detalle_var.set(
            f"Zonas usadas: {z}/7"))

        if resultado["acceso"]:
            self.after(0, lambda r=resultado: self._resultado_ok(r))
        else:
            self.after(0, lambda r=resultado: self._resultado_negado(r))

    def _resultado_ok(self, r):
        self._set_overlay((0,255,136), r["nombre"])
        self.resultado_var.set("ACCESO PERMITIDO")
        self.resultado_label.config(fg=SUCCESS)
        self.candidato_var.set(
            f"{r['nombre']}\nCuenta: {r['numero_cuenta']}\n"
            f"Similitud: {r['similitud_pct']}%")
        self.after(4000, self._lanzar_verificacion)

    def _resultado_negado(self, r):
        self._set_overlay((255,59,92), "Desconocido")
        self.resultado_var.set("ACCESO DENEGADO")
        self.resultado_label.config(fg=DANGER)
        self.candidato_var.set(
            f"Mas parecido a:\n{r['nombre']}\n"
            f"Similitud: {r['similitud_pct']}%")
        self.after(4000, self._lanzar_verificacion)

    def on_close(self):
        self._stop_cam()
        self.destroy()


if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("[ERROR] Falta Pillow. Ejecuta:  pip install Pillow")
        exit(1)
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()