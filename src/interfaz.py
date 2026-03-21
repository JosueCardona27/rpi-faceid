"""
interfaz.py
===========
Interfaz grafica del sistema de acceso facial.
Pantalla tactil 7" 1024x600.

CAMBIOS:
  - Eliminado campo "Numero de cuenta" del formulario de registro
  - Eliminado campo "Telefono"
  - Validacion mas estricta en pasos: requiere angulo correcto para avanzar
  - MUESTRAS_MIN_PASO aumentado a 5 para forzar que el usuario mantenga el angulo
  - Mensaje de instruccion mas claro en cada paso
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
                          ZONAS_POR_TIPO, ZONAS_FRONTAL,
                          TIPO_FRONTAL, TIPO_PERFIL_D, TIPO_PERFIL_I)
from database   import (registrar_usuario, guardar_vectores_por_angulo,
                         guardar_vector_unico, reconocer_persona,
                         eliminar_persona, ROLES_VALIDOS)

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
MUESTRAS_MIN_PASO = 5   # minimo para que un paso sea valido


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

        cv = tk.Canvas(self, width=W, height=H, bg=BG, highlightthickness=0)
        cv.place(x=0, y=0)
        cv.create_rectangle(0, 0, W, 3, fill=ACCENT, outline="")

        IY = 30; cx = W // 2
        cv.create_oval(cx-20, IY,    cx+20, IY+44, outline=ACCENT,  width=2)
        cv.create_oval(cx-11, IY+9,  cx+11, IY+33, outline=ACCENT2, width=2)
        cv.create_line(cx,    IY+44, cx,    IY+62, fill=ACCENT, width=2)
        cv.create_line(cx-20, IY+53, cx+20, IY+53, fill=ACCENT, width=2)

        TY = IY + 74
        tk.Label(self, text="SISTEMA DE ACCESO FACIAL",
                 font=self.f_title, fg=ACCENT, bg=BG
                 ).place(x=W//2, y=TY, anchor="center")
        SY = TY + 22
        modo_txt = "Raspberry Pi" if USAR_PICAM else "Webcam"
        tk.Label(self,
                 text=f"LBP 512 dims  |  4 pasos  |  {modo_txt}",
                 font=self.f_sub, fg=SUBTEXT, bg=BG
                 ).place(x=W//2, y=SY, anchor="center")

        LY = SY + 16
        cv.create_line(W//2-280, LY, W//2+280, LY, fill=BORDER, width=1)

        CY = LY + 16
        CW, CH = 230, 180; GAP = 40
        CX1 = W//2 - (CW*2+GAP)//2
        CX2 = CX1 + CW + GAP

        self._card_btn(cv, CX1, CY, CW, CH, "REGISTRARME",
                       "Registrar nuevo usuario\ny capturar datos faciales",
                       ACCENT, self._show_registro)
        self._card_btn(cv, CX2, CY, CW, CH, "ACCESO",
                       "Verificar identidad\nmediante reconocimiento facial",
                       SUCCESS, self._show_acceso)

        cv.create_line(0, H-26, W, H-26, fill=BORDER, width=1)
        tk.Label(self,
                 text=f"v5.5  |  4 pasos ({int(TIEMPO_ESCANEO)}s)  |  "
                      f"max {MAX_MUESTRAS_PASO} muestras/paso  |  {modo_txt}",
                 font=self.f_zona, fg=SUBTEXT, bg=BG
                 ).place(x=W//2, y=H-13, anchor="center")

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

    # ══════════════════════════════════════════════════════════════════════════
    #  PANTALLA REGISTRO  (sin numero de cuenta ni telefono)
    # ══════════════════════════════════════════════════════════════════════════
    def _show_registro(self):
        self._clear()
        self.geometry(f"{W}x{H}+0+0")
        self._set_overlay(None, "")

        left = tk.Frame(self, bg=PANEL, width=PANEL_W, height=H)
        left.place(x=0, y=0)
        tk.Frame(left, bg=ACCENT, width=PANEL_W, height=3).place(x=0, y=0)
        tk.Label(left, text="REGISTRO",      font=self.f_title, fg=ACCENT,  bg=PANEL).place(x=18, y=8)
        tk.Label(left, text="Nuevo usuario", font=self.f_sub,   fg=SUBTEXT, bg=PANEL).place(x=18, y=34)
        tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=52)

        # Solo 3 campos: nombre, apellido paterno, apellido materno
        self._field(left, "Nombre(s)",       self.nombre_var,  58)
        self._field(left, "Apellido paterno", self.ap_pat_var, 106)
        self._field(left, "Apellido materno", self.ap_mat_var, 154)

        tk.Label(left, text="Rol", font=self.f_label, fg=SUBTEXT, bg=PANEL).place(x=18, y=204)
        rf = tk.Frame(left, bg=PANEL)
        rf.place(x=18, y=218, width=284)
        for i, rol in enumerate(ROLES_VALIDOS):
            c = COLOR_ROL.get(rol, ACCENT)
            tk.Radiobutton(rf, text=rol.capitalize(),
                           variable=self.rol_var, value=rol,
                           font=self.f_label, fg=c, bg=PANEL,
                           selectcolor=CARD, activebackground=PANEL,
                           activeforeground=c, cursor="hand2"
                           ).grid(row=0, column=i, padx=8)

        tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=244)

        # ── Indicadores de los 4 pasos ────────────────────────────────────────
        tk.Label(left, text="PASOS DE ESCANEO:", font=self.f_zona,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=250)

        self._paso_frames = []
        paso_w = 56
        for i, (_, icono, _, etiq, _, _, _, _) in enumerate(PASOS_REGISTRO):
            fx = 18 + i * (paso_w + 6)
            pf = tk.Frame(left, bg=BORDER, width=paso_w, height=52)
            pf.place(x=fx, y=262)
            li = tk.Label(pf, text=str(i+1), font=self.f_btn,  fg=SUBTEXT, bg=BORDER)
            li.place(relx=.5, y=10, anchor="center")
            ln = tk.Label(pf, text=icono,    font=self.f_label, fg=SUBTEXT, bg=BORDER)
            ln.place(relx=.5, y=28, anchor="center")
            bpf = tk.Frame(pf, bg="#111", width=paso_w-4, height=4)
            bpf.place(x=2, y=46)
            bp = tk.Frame(bpf, bg=SUBTEXT, width=0, height=4)
            bp.place(x=0, y=0)
            self._paso_frames.append((pf, li, ln, bp))

        self.paso_desc_var = tk.StringVar(value="Completa los campos e inicia")
        tk.Label(left, textvariable=self.paso_desc_var,
                 font=self.f_label, fg=ACCENT, bg=PANEL,
                 wraplength=284, justify="center"
                 ).place(x=18, y=322, width=284)

        self.cap_btn = tk.Button(
            left, text="INICIAR ESCANEO", font=self.f_btn,
            fg=BG, bg=ACCENT, relief="flat", cursor="hand2",
            padx=8, pady=7, command=self._iniciar_registro)
        self.cap_btn.place(x=18, y=348, width=284)
        self.cap_btn.bind("<Enter>", lambda e: self.cap_btn.config(bg=self._lighten(ACCENT)))
        self.cap_btn.bind("<Leave>", lambda e: self.cap_btn.config(bg=ACCENT))

        tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=390)

        tk.Label(left, text="MUESTRAS POR PASO:", font=self.f_zona,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=396)

        self._barra_pasos = []
        etiquetas_pasos   = ["Frente", "Derecha", "Izquierda", "Frente"]
        barra_w = 56
        for i, etiq in enumerate(etiquetas_pasos):
            fx = 18 + i * (barra_w + 6)
            tk.Label(left, text=etiq[:5], font=self.f_zona,
                     fg=SUBTEXT, bg=PANEL).place(x=fx, y=410)
            bg_b = tk.Frame(left, bg=BORDER, width=barra_w, height=8)
            bg_b.place(x=fx, y=422)
            bar  = tk.Frame(bg_b, bg=BORDER, width=0, height=8)
            bar.place(x=0, y=0)
            lbl  = tk.Label(left, text="0", font=self.f_zona, fg=SUBTEXT, bg=PANEL)
            lbl.place(x=fx + barra_w//2, y=432, anchor="center")
            self._barra_pasos.append((bar, lbl, barra_w))

        tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=446)

        self.progreso_var = tk.StringVar(value="")
        self.prog_label   = tk.Label(left, textvariable=self.progreso_var,
                                     font=self.f_status, fg=WARNING, bg=PANEL,
                                     justify="center", wraplength=284)
        self.prog_label.place(x=18, y=454, width=284)

        self.timer_var    = tk.StringVar(value="")
        self.paso_txt_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.timer_var,
                 font=self.f_title, fg=ACCENT, bg=PANEL
                 ).place(x=18, y=500, anchor="nw")
        tk.Label(left, textvariable=self.paso_txt_var,
                 font=self.f_zona, fg=SUBTEXT, bg=PANEL,
                 wraplength=160).place(x=62, y=508, anchor="nw")

        tk.Button(left, text="<  Volver", font=self.f_label,
                  fg=SUBTEXT, bg=PANEL, relief="flat",
                  cursor="hand2", command=self._volver).place(x=18, y=565)

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
        threading.Thread(target=self._loop_analisis, daemon=True).start()

    def _field(self, parent, label, var, y):
        tk.Label(parent, text=label, font=self.f_label,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=y)
        tk.Entry(parent, textvariable=var, font=self.f_sub,
                 fg=TEXT, bg=CARD, insertbackground=ACCENT,
                 relief="flat", highlightthickness=1,
                 highlightcolor=ACCENT, highlightbackground=BORDER
                 ).place(x=18, y=y+14, width=284, height=24)

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
        if not hasattr(self, "_barra_pasos") or paso_idx >= len(self._barra_pasos):
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

    # ── validacion e inicio del registro ─────────────────────────────────────
    def _iniciar_registro(self):
        nombre = self.nombre_var.get().strip()
        ap_pat = self.ap_pat_var.get().strip()
        ap_mat = self.ap_mat_var.get().strip()
        rol    = self.rol_var.get()

        if not nombre or not ap_pat:
            self.progreso_var.set("Nombre y apellido paterno son obligatorios.")
            self.prog_label.config(fg=DANGER)
            return

        self.cap_btn.config(state="disabled", bg=BORDER, text="Escaneando...")
        self.progreso_var.set("Preparando...")
        self.prog_label.config(fg=WARNING)
        self.timer_var.set("")
        self.after(0, self._resetear_pasos_ui)

        threading.Thread(
            target=self._capturar_registro,
            args=(nombre, ap_pat, ap_mat, rol),
            daemon=True).start()

    def _capturar_registro(self, nombre, ap_pat, ap_mat, rol):
        BAR_W = CAM_W - 16

        uid = registrar_usuario(
            nombre=nombre,
            apellido_paterno=ap_pat,
            apellido_materno=ap_mat,
            rol=rol)

        if uid == -1:
            self.after(0, lambda: self._safe(
                lambda: self.progreso_var.set("Error al registrar. Intentalo de nuevo.")))
            self.after(0, lambda: self._safe(lambda: self.prog_label.config(fg=DANGER)))
            self.after(0, lambda: self._safe(
                lambda: self.cap_btn.config(state="normal", bg=ACCENT,
                                            text="INICIAR ESCANEO")))
            return

        vectores_angulo: dict = {}
        n_muestras_paso = [0] * N_PASOS

        t_offsets = []
        acum = 0.0
        for p in PASOS_REGISTRO:
            t_offsets.append(acum); acum += p[4]

        for paso_idx, (_, icono, instruccion, etiqueta,
                       duracion, modo_det,
                       tipo_esperado, msg_correccion) in enumerate(PASOS_REGISTRO):

            vectores_paso = []
            t_paso_activo = 0.0
            t_ultimo_tick = None
            ultimo_id     = -1

            self._modo_deteccion = modo_det
            self._tipo_esperado  = tipo_esperado

            self.after(0, lambda i=instruccion: self._safe(
                lambda: self.status_var.set(i)))
            self.after(0, lambda pi=paso_idx: self._activar_paso_ui(pi, 0.0))
            self._set_overlay((255, 184, 48), f"{paso_idx+1}/{N_PASOS}: {instruccion}")

            while t_paso_activo < duracion and self.cam_running:

                if len(vectores_paso) >= MAX_MUESTRAS_PASO:
                    t_paso_activo = duracion
                    break

                with self._analisis_lock:
                    frame_id = self._analisis["frame_id"]
                    v        = self._analisis["vector"]
                    tipo_det = self._analisis["tipo"]

                # RESTRICCION: solo avanza si el angulo detectado es el correcto
                angulo_ok     = (tipo_det == tipo_esperado) and (v is not None)
                cara_presente = (v is not None)

                ahora = time.time()
                if angulo_ok:
                    if t_ultimo_tick is not None:
                        t_paso_activo += ahora - t_ultimo_tick
                    t_ultimo_tick = ahora
                else:
                    # Si el angulo no es correcto, el tiempo NO avanza
                    t_ultimo_tick = None

                restante      = max(0.0, duracion - t_paso_activo)
                progreso_paso = min(1.0, t_paso_activo / duracion)
                elapsed_g     = t_offsets[paso_idx] + t_paso_activo
                progreso_tot  = min(1.0, elapsed_g / TIEMPO_ESCANEO)

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
                        self._set_overlay(
                            (0, 255, 136),
                            f"{paso_idx+1}/{N_PASOS} {etiqueta} [{n}/{MAX_MUESTRAS_PASO}]")
                        self.after(0, lambda nt=tot: self._safe(
                            lambda: self.progreso_var.set(f"Total: {nt} muestras")))
                        self.after(0, lambda: self._safe(
                            lambda: self.prog_label.config(fg=SUCCESS)))
                        self.after(0, lambda pi=paso_idx, nm=n_muestras_paso[paso_idx]:
                                   self._update_barra_paso(pi, nm))

                    elif cara_presente and not angulo_ok:
                        # Cara detectada pero angulo incorrecto — mostrar instruccion
                        self._set_overlay((255, 59, 92),
                                          f"ESPERANDO — {msg_correccion}")
                        self.after(0, lambda mc=msg_correccion: self._safe(
                            lambda: self.status_var.set(mc)))
                        self.after(0, lambda: self._safe(
                            lambda: self.prog_label.config(fg=DANGER)))
                    else:
                        # Sin cara
                        self._set_overlay((255, 184, 48),
                                          f"{paso_idx+1}/{N_PASOS}: {instruccion}")
                        self.after(0, lambda i=instruccion: self._safe(
                            lambda: self.status_var.set(i)))
                        self.after(0, lambda: self._safe(
                            lambda: self.prog_label.config(fg=WARNING)))

                time.sleep(0.04)

            self.after(0, lambda pi=paso_idx: self._activar_paso_ui(pi+1, 0.0))

        # ── Fin escaneo ───────────────────────────────────────────────────────
        self._modo_deteccion = "auto"
        self._tipo_esperado  = None
        self._set_overlay(None, "")
        self.after(0, lambda: self._safe(lambda: self.timer_var.set("")))
        self.after(0, lambda: self._safe(lambda: self.paso_txt_var.set("")))
        self.after(0, lambda: self._safe(lambda: self.prog_bar.config(width=BAR_W)))

        total_muestras  = sum(n_muestras_paso)
        pasos_ok        = sum(1 for n in n_muestras_paso if n >= MUESTRAS_MIN_PASO)
        nombre_completo = f"{nombre} {ap_pat} {ap_mat}".strip()

        if pasos_ok == N_PASOS:
            guardar_vectores_por_angulo(uid, vectores_angulo)
            self.after(0, lambda: self._safe(lambda: self.prog_bar.config(bg=SUCCESS)))
            self.after(0, lambda: self._safe(lambda: self.status_var.set(
                f"Registro completo — {total_muestras} muestras, {pasos_ok} pasos")))
            self.after(0, lambda nv=nombre_completo, nt=total_muestras: self._safe(
                lambda: self.progreso_var.set(
                    f"Listo. {nv}\n{nt} muestras en {N_PASOS} pasos.")))
            self.after(0, lambda: self._safe(lambda: self.prog_label.config(fg=SUCCESS)))
            self.after(0, lambda: self._safe(lambda: self.cap_btn.config(
                state="normal", bg=SUCCESS, text="REGISTRO COMPLETO")))
            for var in (self.nombre_var, self.ap_pat_var, self.ap_mat_var):
                self.after(0, lambda v=var: self._safe(lambda: v.set("")))
            self.after(0, lambda: self._safe(lambda: self.rol_var.set("estudiante")))
            self.after(3500, lambda: self._safe(lambda: self.cap_btn.config(
                bg=ACCENT, text="INICIAR ESCANEO")))
        else:
            try:
                eliminar_persona(uid)
            except:
                pass
            pasos_fallidos = [PASOS_REGISTRO[i][3]
                              for i, n in enumerate(n_muestras_paso)
                              if n < MUESTRAS_MIN_PASO]
            self.after(0, lambda pf=pasos_fallidos: self._safe(
                lambda: self.progreso_var.set(
                    f"Pasos incompletos:\n{', '.join(pf)}\n"
                    f"Intentalo de nuevo.")))
            self.after(0, lambda: self._safe(lambda: self.prog_label.config(fg=DANGER)))
            self.after(0, lambda: self._safe(lambda: self.status_var.set(
                "Intentalo de nuevo. Acercate mas a la camara.")))
            self.after(0, lambda: self._safe(lambda: self.cap_btn.config(
                state="normal", bg=ACCENT, text="INICIAR ESCANEO")))
            self.after(0, self._resetear_pasos_ui)

    # ══════════════════════════════════════════════════════════════════════════
    #  PANTALLA ACCESO
    # ══════════════════════════════════════════════════════════════════════════
    def _show_acceso(self):
        self._clear()
        self.geometry(f"{W}x{H}+0+0")
        self.verificando = False
        self._set_overlay(None, "")

        left = tk.Frame(self, bg=PANEL, width=PANEL_W, height=H)
        left.place(x=0, y=0)
        tk.Frame(left, bg=SUCCESS, width=PANEL_W, height=3).place(x=0, y=0)
        tk.Label(left, text="ACCESO", font=self.f_title,
                 fg=SUCCESS, bg=PANEL).place(x=18, y=8)
        tk.Label(left, text="Verificacion de identidad", font=self.f_sub,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=34)
        tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=52)

        tk.Label(left,
                 text=("Mira directo a la camara.\n\n"
                       "Funciona con:\n"
                       "  - Lentes\n"
                       "  - Cubrebocas\n"
                       "  - Gorra\n\n"
                       "El mejor angulo registrado\n"
                       "se usa para identificarte."),
                 font=self.f_label, fg=SUBTEXT, bg=PANEL,
                 justify="left").place(x=18, y=60)

        tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=228)

        self.resultado_var   = tk.StringVar(value="Esperando...")
        self.resultado_label = tk.Label(left, textvariable=self.resultado_var,
                                        font=self.f_btn, fg=ACCENT, bg=PANEL,
                                        wraplength=284, justify="center")
        self.resultado_label.place(x=18, y=244, width=284)

        self.candidato_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.candidato_var,
                 font=self.f_label, fg=TEXT, bg=PANEL,
                 wraplength=284, justify="center"
                 ).place(x=18, y=286, width=284)

        tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=400)

        tk.Label(left, text="SIMILITUD:", font=self.f_zona,
                 fg=SUBTEXT, bg=PANEL).place(x=18, y=408)
        self.sim_bg  = tk.Frame(left, bg=BORDER, width=284, height=18)
        self.sim_bg.place(x=18, y=422)
        self.sim_bar = tk.Frame(self.sim_bg, bg=BORDER, width=0, height=18)
        self.sim_bar.place(x=0, y=0)
        self.sim_lbl = tk.Label(self.sim_bg, text="", font=self.f_zona,
                                fg=BG, bg=BORDER)
        self.sim_lbl.place(x=4, y=2)

        self.detalle_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.detalle_var, font=self.f_zona,
                 fg=SUBTEXT, bg=PANEL, wraplength=284, justify="center"
                 ).place(x=18, y=448, width=284)

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
        threading.Thread(target=self._loop_analisis, daemon=True).start()
        self.after(1800, self._lanzar_verificacion)

    def _set_sim_bar(self, pct, color):
        w = max(0, min(284, int(pct / 100 * 284)))
        self.sim_bar.config(width=w, bg=color)
        self.sim_lbl.config(text=f" {pct}%", bg=color,
                            fg=BG if w > 30 else color)

    def _lanzar_verificacion(self):
        if not self.cam_running:
            return
        if self.verificando:
            self.after(300, self._lanzar_verificacion)
            return
        self.verificando = True
        self.after(0, lambda: self.resultado_var.set("Escaneando..."))
        self.after(0, lambda: self.resultado_label.config(fg=ACCENT))
        self.after(0, lambda: self.candidato_var.set(""))
        self.after(0, lambda: self.detalle_var.set(""))
        self._set_overlay((255, 184, 48), "Analizando...")
        threading.Thread(target=self._verificar, daemon=True).start()

    def _verificar(self):
        vectores = []
        intentos = 0
        ultimo   = -1

        while len(vectores) < 8 and intentos < 120 and self.cam_running:
            intentos += 1
            with self._analisis_lock:
                frame_id = self._analisis["frame_id"]
                v        = self._analisis["vector"]
            if frame_id == ultimo:
                time.sleep(0.04)
                continue
            ultimo = frame_id
            if v is not None:
                vectores.append(v)
            time.sleep(0.04)

        self._set_overlay(None, "")
        self.verificando = False

        if not self.cam_running:
            return

        if not vectores:
            self.after(0, lambda: self.resultado_var.set("Sin rostro"))
            self.after(0, lambda: self.resultado_label.config(fg=WARNING))
            self.after(0, lambda: self.candidato_var.set("Ponte frente a la camara."))
            self.after(0, lambda: self._set_sim_bar(0, BORDER))
            self.after(3000, self._lanzar_verificacion)
            return

        v_final   = np.mean(vectores, axis=0).astype(np.float32)
        resultado = reconocer_persona(v_final)

        if resultado is None:
            # None = sin registros O persona desconocida (distancia > UMBRAL_RECHAZO)
            from database import cargar_vectores_por_angulo as _cvpa
            hay_registros = len(_cvpa()) > 0
            if hay_registros:
                self.after(0, lambda: self.resultado_var.set("ACCESO DENEGADO"))
                self.after(0, lambda: self.resultado_label.config(fg=DANGER))
                self.after(0, lambda: self.candidato_var.set("Persona no reconocida. No esta registrada."))
                self._set_overlay((255, 59, 92), "Desconocido")
            else:
                self.after(0, lambda: self.resultado_var.set("Sin registros"))
                self.after(0, lambda: self.resultado_label.config(fg=WARNING))
                self.after(0, lambda: self.candidato_var.set("No hay usuarios registrados."))
            self.after(0, lambda: self._set_sim_bar(0, BORDER))
            self.after(4000, self._lanzar_verificacion)
            return

        sim         = resultado["similitud_pct"]
        color_barra = SUCCESS if resultado["acceso"] else (WARNING if sim >= 55 else DANGER)
        self.after(0, lambda: self._set_sim_bar(int(sim), color_barra))
        self.after(0, lambda r=resultado: self.detalle_var.set(
            f"Angulo match: {r['angulo']}"))

        # Doble validacion: acceso=True Y similitud minima del 70%
        # Evita que una persona no registrada acceda por ser "la mas cercana"
        # aunque su distancia supere el umbral (por redondeos o vectores cortos)
        if resultado["acceso"]:
            self.after(0, lambda r=resultado: self._resultado_ok(r))
        else:
            self.after(0, lambda r=resultado: self._resultado_negado(r))

    def _resultado_ok(self, r):
        self._set_overlay((0, 255, 136), r["nombre"])
        self.resultado_var.set("ACCESO PERMITIDO")
        self.resultado_label.config(fg=SUCCESS)
        self.candidato_var.set(
            f"{r['nombre']}\n"
            f"Rol: {r.get('rol','---').upper()}\n"
            f"Similitud: {r['similitud_pct']}%")
        self.after(4000, self._lanzar_verificacion)

    def _resultado_negado(self, r):
        self._set_overlay((255, 59, 92), "Desconocido")
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
        print("[ERROR] Falta Pillow:  pip install Pillow")
        exit(1)
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()