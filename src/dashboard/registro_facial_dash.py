"""
registro_facial_dash.py
=======================
Diálogo de captura biométrica embebido en el Dashboard.

Reemplaza la invocación de subprocess a interfaz.py.
El usuario YA está creado en la BD (uid conocido) antes de abrir este diálogo.

Uso:
    from registro_facial_dash import CapturaBiometricaDialog

    CapturaBiometricaDialog(
        parent     = dash.root,
        uid        = usuario_id,   # ID ya guardado en la BD
        datos      = {             # dict con los datos del usuario
            "nombre": "...", "ap": "...", "am": "...",
            "cuenta": "...", "rol": "...",
            # estudiante:
            "grado": "...", "grupo": "...",
            # maestro/admin:
            "correo": "...",
        },
        on_success = callback,     # llamado al completar la captura
        on_cancel  = callback,     # llamado al cancelar (uid eliminado)
    )
"""

import tkinter as tk
from tkinter import messagebox
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk

# ── Detección de cámara ──────────────────────────────────────────────
USAR_PICAM = False
try:
    from picamera2 import Picamera2
    _t = Picamera2(); _t.close(); del _t
    USAR_PICAM = True
    print("[CAM-DASH] Picamera2 disponible")
except Exception:
    print("[CAM-DASH] Usando webcam OpenCV")

# ── Motor facial ─────────────────────────────────────────────────────
from face_engine import (
    extraer_caracteristicas, dibujar_overlay,
    TIPO_FRONTAL, TIPO_PERFIL_D, TIPO_PERFIL_I,
)

# ── Base de datos ────────────────────────────────────────────────────
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from database import (
    guardar_vectores_por_angulo,
    verificar_duplicado_facial,
    eliminar_persona,
)

# ── Tema del dashboard ───────────────────────────────────────────────
from dash_theme import BG, CARD, CARD2, ACCENT, RED, T1, T2, T3, BORDER, iniciales

_SUCCESS = "#22C55E"
_DANGER  = "#EF4444"
_WARNING = "#F59E0B"
_PANEL   = "#161B26"

# ── Dimensiones del diálogo ──────────────────────────────────────────
_W       = 950
_H       = 630
_LEFT_W  = 290
_CAM_W   = _W - _LEFT_W      # 660
_CAM_H   = 360

# ── Pasos de captura (idénticos a interfaz.py v5.5) ──────────────────
# Formato: (instruccion, etiqueta, duracion_s, modo_det, tipo_esperado, msg_correccion)
_PASOS = [
    ("Mira directo a la cámara",
     "FRENTE",    12.0, "frontal", TIPO_FRONTAL,  "Mira directo a la cámara"),
    ("Gira tu cabeza a la IZQUIERDA",
     "IZQUIERDA", 10.0, "perfil",  TIPO_PERFIL_D, "Gira más a tu izquierda"),
    ("Gira tu cabeza a la DERECHA",
     "DERECHA",   10.0, "perfil",  TIPO_PERFIL_I, "Gira más a tu derecha"),
    ("Vuelve al frente (confirmación)",
     "FRENTE",    12.0, "frontal", TIPO_FRONTAL,  "Mira directo a la cámara"),
]
_N_PASOS      = len(_PASOS)
_MAX_MUESTRAS = 50
_MUESTRAS_MIN = 10
_TIEMPO_TOTAL = sum(p[2] for p in _PASOS)   # segundos totales


# ════════════════════════════════════════════════════════════════════
class CapturaBiometricaDialog(tk.Toplevel):
    """
    Ventana modal con captura facial de 4 pasos integrada al dashboard.

    Botones disponibles:
      · ← Volver          → cierra el diálogo; el registro queda sin biometría
      · ✕ Cancelar        → elimina el registro de la BD y cierra
      · ▶ Iniciar Captura → lanza el escaneo biométrico de 4 ángulos
    """

    def __init__(self, parent: tk.Tk | tk.Toplevel,
                 uid: int,
                 datos: dict,
                 on_success=None,
                 on_cancel=None):
        super().__init__(parent)
        self.uid        = uid
        self.datos      = datos
        self.on_success = on_success
        self.on_cancel  = on_cancel

        # ── Estado cámara ─────────────────────────────────────────────
        self.picam2      = None
        self._cap        = None
        self.cam_running = False
        self._capturando = False

        self._frame_actual = None
        self._frame_lock   = threading.Lock()

        self._analisis = {"vector": None, "coords": None,
                          "frame_id": -1, "tipo": None}
        self._analisis_lock = threading.Lock()

        self._ov_color = None
        self._ov_texto = ""
        self._ov_lock  = threading.Lock()

        self._modo_deteccion = "auto"
        self._tipo_esperado  = None

        # ── Ventana ───────────────────────────────────────────────────
        self.title("Registro Facial Biométrico")
        self.resizable(False, False)
        self.configure(bg=BG)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._volver)

        # Centrar sobre el padre
        self.update_idletasks()
        px = parent.winfo_rootx() + parent.winfo_width()  // 2 - _W // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - _H // 2
        self.geometry(f"{_W}x{_H}+{px}+{py}")

        self._build()

        # ── Iniciar cámara y threads ──────────────────────────────────
        try:
            self._start_cam()
            self.cam_running = True
            threading.Thread(target=self._loop_camara,  daemon=True).start()
            threading.Thread(target=self._loop_analisis, daemon=True).start()
        except RuntimeError as e:
            messagebox.showerror("Error de cámara", str(e), parent=self)

    # ════════════════════════════════════════════════════════════════
    # CÁMARA
    # ════════════════════════════════════════════════════════════════
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
                raise RuntimeError("No se pudo abrir ninguna cámara.")
            for _ in range(5):
                self._cap.read()

    def _stop_cam(self):
        self.cam_running = False
        time.sleep(0.25)
        if USAR_PICAM:
            if self.picam2:
                try:    self.picam2.close()
                except: pass
                self.picam2 = None
        else:
            if self._cap:
                try:    self._cap.release()
                except: pass
                self._cap = None

    def _leer_frame(self):
        try:
            if USAR_PICAM:
                frame = self.picam2.capture_array()
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, raw = self._cap.read()
            return raw if ret else None
        except:
            return None

    def _loop_camara(self):
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
                    (0, 201, 167) if vector is not None else (80, 80, 80))
                t = ov_texto or ("Detectado" if vector is not None
                                 else "Buscando...")
                vis = dibujar_overlay(vis, coords, c, t, tipo=tipo)

            imgtk = self._to_imgtk(vis)
            self.after(0, self._mostrar_frame, imgtk)
            time.sleep(0.033)

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
                frame,
                modo=self._modo_deteccion,
                tipo_esperado=self._tipo_esperado,
            )
            with self._analisis_lock:
                self._analisis["vector"]   = vector
                self._analisis["coords"]   = coords
                self._analisis["frame_id"] = frame_id
                self._analisis["tipo"]     = tipo

    def _mostrar_frame(self, imgtk):
        if not self.cam_running:
            return
        try:
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)
        except:
            pass

    @staticmethod
    def _to_imgtk(frame) -> ImageTk.PhotoImage:
        h0, w0 = frame.shape[:2]
        r  = min((_CAM_W - 32) / w0, (_CAM_H - 16) / h0)
        fr = cv2.resize(frame, (int(w0 * r), int(h0 * r)))
        return ImageTk.PhotoImage(
            image=Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))

    def _set_overlay(self, color, texto: str = ""):
        with self._ov_lock:
            self._ov_color = color
            self._ov_texto = texto

    # ════════════════════════════════════════════════════════════════
    # CONSTRUCCIÓN DE UI
    # ════════════════════════════════════════════════════════════════
    def _build(self):
        d = self.datos

        # ════════════════ PANEL IZQUIERDO — Datos + Botones ══════════
        left = tk.Frame(self, bg=_PANEL, width=_LEFT_W)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # ── Encabezado ────────────────────────────────────────────────
        tk.Label(
            left, text="📋  Datos del registro",
            bg=_PANEL, fg=T2, font=("Arial", 8, "bold"),
            anchor="w",
        ).pack(fill="x", padx=16, pady=(18, 6))

        # ── Tarjeta de datos ──────────────────────────────────────────
        card = tk.Frame(left, bg=CARD, padx=12, pady=10)
        card.pack(fill="x", padx=12, pady=(0, 8))

        ini = iniciales(d.get("nombre", ""), d.get("ap", ""))
        tk.Label(
            card, text=ini, bg=ACCENT, fg=BG,
            font=("Arial", 16, "bold"), width=3, height=1,
        ).pack(pady=(4, 8))

        nombre_completo = (f"{d.get('nombre', '')} {d.get('ap', '')} "
                           f"{d.get('am', '')}").strip()
        tk.Label(
            card, text=nombre_completo,
            bg=CARD, fg=T1, font=("Arial", 10, "bold"),
            wraplength=_LEFT_W - 48,
        ).pack()

        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", pady=(8, 4))

        def _fila(lbl: str, val):
            row = tk.Frame(card, bg=CARD)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=lbl, bg=CARD, fg=T3,
                     font=("Arial", 8), width=9, anchor="w").pack(side="left")
            tk.Label(row, text=str(val) if val else "—",
                     bg=CARD, fg=T2, font=("Arial", 8),
                     anchor="w").pack(side="left", padx=4)

        _fila("Cuenta:",  d.get("cuenta"))
        _fila("Rol:",     (d.get("rol") or "").capitalize())

        if d.get("correo"):
            _fila("Correo:", d["correo"])
        if d.get("grado"):
            _fila("Grado:",  d["grado"])
            _fila("Grupo:",  d.get("grupo") or "—")

        # ── Nota informativa ─────────────────────────────────────────
        info = tk.Frame(left, bg="#1a2035", padx=10, pady=8)
        info.pack(fill="x", padx=12, pady=(4, 4))
        tk.Label(
            info, text="📷  Captura de 4 ángulos",
            bg="#1a2035", fg=ACCENT,
            font=("Arial", 8, "bold"), anchor="w",
        ).pack(fill="x")
        tk.Label(
            info,
            text="Frente → Izquierda → Derecha → Frente\n"
                 "Sigue las instrucciones en pantalla.",
            bg="#1a2035", fg=T2, font=("Arial", 7),
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        # ── Etiqueta de estado ────────────────────────────────────────
        self._status_var = tk.StringVar(
            value="Presiona Iniciar para comenzar")
        tk.Label(
            left, textvariable=self._status_var,
            bg=_PANEL, fg=_WARNING, font=("Arial", 8),
            wraplength=_LEFT_W - 24, justify="center",
        ).pack(padx=12, pady=(6, 4))

        # ── Separador ─────────────────────────────────────────────────
        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", padx=12, pady=4)

        # ── Botones ───────────────────────────────────────────────────
        btn_area = tk.Frame(left, bg=_PANEL)
        btn_area.pack(side="bottom", fill="x", padx=12, pady=14)

        self._btn_iniciar = tk.Button(
            btn_area,
            text="▶  Iniciar Captura Facial",
            command=self._iniciar_captura,
            bg=ACCENT, fg=BG, relief="flat",
            font=("Arial", 9, "bold"),
            padx=10, pady=8, cursor="hand2",
        )
        self._btn_iniciar.pack(fill="x", pady=(0, 6))

        tk.Button(
            btn_area, text="← Volver",
            command=self._volver,
            bg=CARD2, fg=T2, relief="flat",
            font=("Arial", 9), padx=10, pady=6,
            cursor="hand2",
        ).pack(fill="x", pady=(0, 4))

        tk.Button(
            btn_area, text="✕  Cancelar registro",
            command=self._cancelar,
            bg=RED, fg="white", relief="flat",
            font=("Arial", 9), padx=10, pady=6,
            cursor="hand2",
        ).pack(fill="x")

        # ════════════════ PANEL DERECHO — Cámara ═════════════════════
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(
            right, text="CAPTURA BIOMÉTRICA",
            bg=BG, fg=ACCENT, font=("Arial", 10, "bold"),
        ).place(x=12, y=10)

        # Fondo del área de cámara
        cam_bg = tk.Frame(right, bg="#080A0F",
                          width=_CAM_W - 16, height=_CAM_H)
        cam_bg.place(x=8, y=36)
        cam_bg.pack_propagate(False)

        self.cam_label = tk.Label(cam_bg, bg="#080A0F")
        self.cam_label.place(x=0, y=0, relwidth=1, relheight=1)

        # ── Indicadores de paso ───────────────────────────────────────
        _STEP_Y   = 36 + _CAM_H + 10
        _R        = 14
        _LABELS   = ["F", "I", "D", "F"]
        _NAMES    = ["Frente", "Izq.", "Der.", "Frente"]
        _GAP      = (_CAM_W - 16 - 4 * _R * 2) // 5

        paso_cv = tk.Canvas(right, width=_CAM_W - 16, height=70,
                             bg=BG, highlightthickness=0)
        paso_cv.place(x=8, y=_STEP_Y)

        # Línea de fondo entre círculos
        _ly = _R
        paso_cv.create_line(
            _R * 2 + _GAP, _ly,
            _CAM_W - 16 - _R * 2 - _GAP, _ly,
            fill=BORDER, width=2,
        )

        self._paso_canvas  = paso_cv
        self._paso_cx      = []
        self._paso_circles = []
        self._paso_nums    = []
        self._paso_prog    = []

        for i in range(_N_PASOS):
            cx = _GAP + _R + i * (_R * 2 + _GAP)
            cy = _R
            cid = paso_cv.create_oval(
                cx - _R, cy - _R, cx + _R, cy + _R,
                fill=BORDER, outline=BORDER, width=2,
            )
            nid = paso_cv.create_text(cx, cy, text=_LABELS[i],
                                       font=("Arial", 9, "bold"), fill=T3)
            paso_cv.create_text(cx, cy + _R + 10, text=_NAMES[i],
                                 font=("Arial", 7), fill=T3)
            cnt = paso_cv.create_text(cx, cy + _R + 22,
                                       text="0", font=("Arial", 7), fill=T3)
            self._paso_cx.append(cx)
            self._paso_circles.append(cid)
            self._paso_nums.append(nid)
            self._paso_prog.append(cnt)

        # Instrucción del paso actual
        self._paso_desc_var = tk.StringVar(value="")
        tk.Label(
            right, textvariable=self._paso_desc_var,
            bg=BG, fg=ACCENT, font=("Arial", 9),
            wraplength=_CAM_W - 24, justify="center",
        ).place(x=8, y=_STEP_Y + 56, width=_CAM_W - 16)

        # ── Barra de progreso global ──────────────────────────────────
        _BAR_Y = _STEP_Y + 84
        _BAR_W = _CAM_W - 32

        self._prog_bar_bg = tk.Frame(right, bg=BORDER,
                                      width=_BAR_W, height=5)
        self._prog_bar_bg.place(x=16, y=_BAR_Y)
        self._prog_bar = tk.Frame(self._prog_bar_bg,
                                   bg=ACCENT, width=0, height=5)
        self._prog_bar.place(x=0, y=0)

        # Texto de progreso / errores
        self._prog_txt = tk.StringVar(value="")
        self._prog_lbl = tk.Label(
            right, textvariable=self._prog_txt,
            bg=BG, fg=T2, font=("Arial", 8, "bold"),
            justify="center",
        )
        self._prog_lbl.place(x=8, y=_BAR_Y + 10, width=_CAM_W - 16)

        # Temporizador del paso actual
        self._timer_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self._timer_var,
                 bg=BG, fg=ACCENT, font=("Arial", 14, "bold"),
                 ).place(x=16, y=_BAR_Y + 28, anchor="nw")

        self._BAR_W = _BAR_W   # guardado para usarlo en el thread

    # ════════════════════════════════════════════════════════════════
    # INDICADORES VISUALES DE PASO
    # ════════════════════════════════════════════════════════════════
    def _activar_paso(self, paso_idx: int):
        try:
            cv = self._paso_canvas
            for i, (cid, nid) in enumerate(
                    zip(self._paso_circles, self._paso_nums)):
                if i < paso_idx:
                    cv.itemconfig(cid, fill=_SUCCESS, outline=_SUCCESS)
                    cv.itemconfig(nid, fill=BG)
                elif i == paso_idx:
                    cv.itemconfig(cid, fill=ACCENT, outline=ACCENT)
                    cv.itemconfig(nid, fill=BG)
                else:
                    cv.itemconfig(cid, fill=BORDER, outline=BORDER)
                    cv.itemconfig(nid, fill=T3)
        except:
            pass

    def _reset_pasos(self):
        try:
            cv = self._paso_canvas
            for cid, nid, cnt in zip(self._paso_circles,
                                      self._paso_nums,
                                      self._paso_prog):
                cv.itemconfig(cid, fill=BORDER, outline=BORDER)
                cv.itemconfig(nid, fill=T3)
                cv.itemconfig(cnt, text="0", fill=T3)
        except:
            pass

    def _update_paso_count(self, paso_idx: int, n: int):
        try:
            pct = min(1.0, n / _MAX_MUESTRAS)
            col = _SUCCESS if pct >= 1.0 else ACCENT if pct > 0.4 else _WARNING
            self._paso_canvas.itemconfig(
                self._paso_prog[paso_idx], text=str(n), fill=col)
        except:
            pass

    # ════════════════════════════════════════════════════════════════
    # CAPTURA BIOMÉTRICA (corre en un thread aparte)
    # ════════════════════════════════════════════════════════════════
    def _iniciar_captura(self):
        if self._capturando:
            return
        self._capturando = True
        self._btn_iniciar.config(state="disabled", bg=BORDER,
                                  text="Escaneando...")
        self._prog_txt.set("Preparando escaneo...")
        self._prog_lbl.config(fg=_WARNING)
        self._timer_var.set("")
        self.after(0, self._reset_pasos)
        threading.Thread(target=self._run_captura, daemon=True).start()

    def _run_captura(self):
        BAR_W = self._BAR_W
        vectores_angulo: dict  = {}
        n_muestras_paso        = [0] * _N_PASOS
        t_offsets: list[float] = []
        acum = 0.0
        for p in _PASOS:
            t_offsets.append(acum)
            acum += p[2]

        for paso_idx, (instruccion, etiqueta, duracion,
                        modo_det, tipo_esperado,
                        msg_correccion) in enumerate(_PASOS):

            vectores_paso: list = []
            t_paso_activo = 0.0
            t_ultimo_tick = None
            ultimo_id     = -1

            self._modo_deteccion = modo_det
            self._tipo_esperado  = tipo_esperado

            self.after(0, lambda i=instruccion:
                       self._status_var.set(i))
            self.after(0, lambda d=instruccion:
                       self._paso_desc_var.set(d))
            self.after(0, lambda pi=paso_idx:
                       self._activar_paso(pi))
            self._set_overlay((255, 184, 48),
                               f"{paso_idx + 1}/{_N_PASOS}: {instruccion}")

            while t_paso_activo < duracion and self.cam_running:
                if len(vectores_paso) >= _MAX_MUESTRAS:
                    break

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
                progreso_tot = min(
                    1.0,
                    (t_offsets[paso_idx] + t_paso_activo) / _TIEMPO_TOTAL,
                )

                self.after(0, lambda s=int(restante) + 1:
                           self._timer_var.set(f"{s}s"))
                self.after(0, lambda w=int(progreso_tot * BAR_W):
                           self._prog_bar.config(width=w))

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
                            (0, 201, 167),
                            f"{paso_idx + 1}/{_N_PASOS} {etiqueta} "
                            f"[{n}/{_MAX_MUESTRAS}]",
                        )
                        self.after(0, lambda nt=tot:
                                   self._prog_txt.set(
                                       f"Total: {nt} muestras capturadas"))
                        self.after(0, lambda:
                                   self._prog_lbl.config(fg=_SUCCESS))
                        self.after(0, lambda pi=paso_idx,
                                          nm=n_muestras_paso[paso_idx]:
                                   self._update_paso_count(pi, nm))

                    elif cara_presente:
                        self._set_overlay((255, 59, 92),
                                          f"ESPERANDO — {msg_correccion}")
                        self.after(0, lambda mc=msg_correccion:
                                   self._status_var.set(mc))
                        self.after(0, lambda:
                                   self._prog_lbl.config(fg=_DANGER))
                    else:
                        self._set_overlay((255, 184, 48),
                                          f"{paso_idx + 1}/{_N_PASOS}: "
                                          f"{instruccion}")
                        self.after(0, lambda i=instruccion:
                                   self._status_var.set(i))
                        self.after(0, lambda:
                                   self._prog_lbl.config(fg=_WARNING))

                time.sleep(0.04)

            # ── Verificación anti-duplicado al finalizar el paso ──────
            if (self.cam_running
                    and n_muestras_paso[paso_idx] >= _MUESTRAS_MIN):
                snapshot = {
                    ang: np.mean(d["vectores"], axis=0).astype(np.float32)
                    for ang, d in vectores_angulo.items()
                    if d["vectores"]
                }
                dup = verificar_duplicado_facial(snapshot,
                                                 excluir_id=self.uid)
                if dup:
                    nombre_dup = dup.get("nombre", "Persona desconocida")
                    self._set_overlay((255, 59, 92),
                                      f"YA REGISTRADO\n{nombre_dup}")
                    self.after(0, lambda nd=nombre_dup:
                               self._on_duplicado(nd))
                    return

            # Marcar el círculo del paso como completado
            self.after(0, lambda pi=paso_idx:
                       self._activar_paso(pi + 1))

        # ── Fin del escaneo ───────────────────────────────────────────
        self._modo_deteccion = "auto"
        self._tipo_esperado  = None
        self._set_overlay(None, "")
        self.after(0, lambda: self._timer_var.set(""))
        self.after(0, lambda: self._prog_bar.config(width=BAR_W))

        pasos_ok = sum(1 for n in n_muestras_paso if n >= _MUESTRAS_MIN)
        total    = sum(n_muestras_paso)

        if pasos_ok == _N_PASOS:
            # Verificación final anti-duplicado
            snapshot_final = {
                ang: np.mean(d["vectores"], axis=0).astype(np.float32)
                for ang, d in vectores_angulo.items()
                if d["vectores"]
            }
            dup = verificar_duplicado_facial(snapshot_final,
                                             excluir_id=self.uid)
            if dup:
                nombre_dup = dup.get("nombre", "Persona desconocida")
                self.after(0, lambda nd=nombre_dup: self._on_duplicado(nd))
                return

            guardar_vectores_por_angulo(self.uid, vectores_angulo)
            self.after(0, lambda nt=total: self._on_exito(nt))
        else:
            self.after(0, lambda: self._on_incompleto(n_muestras_paso))

    # ════════════════════════════════════════════════════════════════
    # CALLBACKS FINALES (ejecutados en el main thread vía .after())
    # ════════════════════════════════════════════════════════════════
    def _on_exito(self, total_muestras: int):
        self._status_var.set(f"¡Registro completo! — {total_muestras} muestras")
        self._paso_desc_var.set("")
        self._prog_lbl.config(fg=_SUCCESS)
        self._prog_txt.set("✓  Biometría guardada correctamente.")
        self._btn_iniciar.config(
            state="disabled", bg=_SUCCESS, text="✓  Completado")
        self.after(1800, self._cerrar_exito)

    def _cerrar_exito(self):
        self._stop_cam()
        self.grab_release()
        self.destroy()
        if self.on_success:
            self.on_success()

    def _on_incompleto(self, n_muestras_paso: list):
        pasos_fallidos = [i + 1 for i, n in enumerate(n_muestras_paso)
                          if n < _MUESTRAS_MIN]
        self._status_var.set(
            f"Captura incompleta — pasos sin suficientes muestras: "
            f"{pasos_fallidos}")
        self._prog_lbl.config(fg=_DANGER)
        self._prog_txt.set("Intenta de nuevo con mejor iluminación.")
        self._btn_iniciar.config(
            state="normal", bg=ACCENT, text="▶  Reintentar")
        self._capturando = False

    def _on_duplicado(self, nombre_dup: str):
        self._status_var.set(
            f"Rostro ya registrado como:\n{nombre_dup}")
        self._prog_lbl.config(fg=_DANGER)
        self._prog_txt.set(
            "El rostro coincide con un registro existente.\n"
            "Verifica los datos o cancela el registro.")
        self._btn_iniciar.config(
            state="normal", bg=RED, text="Intentar de nuevo")
        self.after(5000, lambda: self._btn_iniciar.config(
            bg=ACCENT, text="▶  Iniciar Captura Facial"))
        self._capturando = False

    # ════════════════════════════════════════════════════════════════
    # ACCIONES DE BOTONES
    # ════════════════════════════════════════════════════════════════
    def _volver(self):
        """Cierra sin eliminar. El registro queda sin biometría."""
        if self._capturando:
            if not messagebox.askyesno(
                    "Captura en progreso",
                    "¿Deseas salir?\n\n"
                    "El registro quedará guardado pero SIN biometría.\n"
                    "Podrás completarla más tarde desde el dashboard.",
                    parent=self,
            ):
                return
        self._stop_cam()
        self.grab_release()
        self.destroy()
        # No se llama a on_success ni on_cancel → el registro existe sin biometría

    def _cancelar(self):
        """Elimina el registro de la BD y cierra."""
        if not messagebox.askyesno(
                "Cancelar registro",
                "¿Seguro que deseas cancelar?\n\n"
                "El registro completo será eliminado de la base de datos.",
                parent=self,
        ):
            return
        try:
            eliminar_persona(self.uid)
            print(f"[CAPTURA] Registro uid={self.uid} eliminado por cancelación.")
        except Exception as e:
            print(f"[CAPTURA] Error al eliminar uid={self.uid}: {e}")
        self._stop_cam()
        self.grab_release()
        self.destroy()
        if self.on_cancel:
            self.on_cancel()