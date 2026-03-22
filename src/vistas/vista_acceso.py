"""
Pantalla de acceso: verificación facial en tiempo real.
Se invoca desde app_base.py como _show_acceso().
"""

import tkinter as tk
import threading
import time
import numpy as np

from vistas.app_base import (BG, PANEL, CARD, ACCENT, SUCCESS, DANGER, WARNING,
                       TEXT, SUBTEXT, BORDER,
                       W, H, PANEL_W, CAM_W)
from database import reconocer_persona, cargar_vectores_por_angulo


def show_acceso(app):
    """Construye y muestra la pantalla de acceso/verificación."""
    app._clear()
    app.geometry(f"{W}x{H}+0+0")
    app.verificando = False
    app._set_overlay(None, "")

    left = tk.Frame(app, bg=PANEL, width=PANEL_W, height=H)
    left.place(x=0, y=0)
    tk.Frame(left, bg=SUCCESS, width=PANEL_W, height=3).place(x=0, y=0)
    tk.Label(left, text="ACCESO", font=app.f_title,
             fg=SUCCESS, bg=PANEL).place(x=18, y=8)
    tk.Label(left, text="Verificacion de identidad", font=app.f_sub,
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
             font=app.f_label, fg=SUBTEXT, bg=PANEL,
             justify="left").place(x=18, y=60)

    tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=228)

    app.resultado_var   = tk.StringVar(value="Esperando...")
    app.resultado_label = tk.Label(left, textvariable=app.resultado_var,
                                    font=app.f_btn, fg=ACCENT, bg=PANEL,
                                    wraplength=284, justify="center")
    app.resultado_label.place(x=18, y=244, width=284)

    app.candidato_var = tk.StringVar(value="")
    tk.Label(left, textvariable=app.candidato_var,
             font=app.f_label, fg=TEXT, bg=PANEL,
             wraplength=284, justify="center"
             ).place(x=18, y=286, width=284)

    tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=400)

    tk.Label(left, text="SIMILITUD:", font=app.f_zona,
             fg=SUBTEXT, bg=PANEL).place(x=18, y=408)
    app.sim_bg  = tk.Frame(left, bg=BORDER, width=284, height=18)
    app.sim_bg.place(x=18, y=422)
    app.sim_bar = tk.Frame(app.sim_bg, bg=BORDER, width=0, height=18)
    app.sim_bar.place(x=0, y=0)
    app.sim_lbl = tk.Label(app.sim_bg, text="", font=app.f_zona,
                            fg=BG, bg=BORDER)
    app.sim_lbl.place(x=4, y=2)

    app.detalle_var = tk.StringVar(value="")
    tk.Label(left, textvariable=app.detalle_var, font=app.f_zona,
             fg=SUBTEXT, bg=PANEL, wraplength=284, justify="center"
             ).place(x=18, y=448, width=284)

    tk.Button(left, text="<  Volver", font=app.f_label,
              fg=SUBTEXT, bg=PANEL, relief="flat",
              cursor="hand2", command=app._volver).place(x=18, y=556)

    right = tk.Frame(app, bg=BG, width=CAM_W, height=H)
    right.place(x=PANEL_W, y=0)
    app.cam_label = tk.Label(right, bg="#080A0F")
    app.cam_label.place(x=8, y=5, width=CAM_W-16, height=H-10)

    app._start_cam()
    app.cam_running = True
    app._ultima_cara_t = time.time()
    threading.Thread(target=app._loop_camara,
                     kwargs={"max_w": CAM_W-16, "max_h": H-10},
                     daemon=True).start()
    threading.Thread(target=app._loop_analisis, daemon=True).start()
    threading.Thread(target=lambda: _monitor_cara(app), daemon=True).start()
    app.after(1800, lambda: _lanzar_verificacion(app))


# ── Lógica de verificación ────────────────────────────────────────────────────

def _set_sim_bar(app, pct, color):
    w = max(0, min(284, int(pct / 100 * 284)))
    app.sim_bar.config(width=w, bg=color)
    app.sim_lbl.config(text=f" {pct}%", bg=color,
                        fg=BG if w > 30 else color)


def _lanzar_verificacion(app):
    if not app.cam_running:
        return
    if app.verificando:
        app.after(300, lambda: _lanzar_verificacion(app))
        return
    app.verificando = True
    app.after(0, lambda: app.resultado_var.set("Escaneando..."))
    app.after(0, lambda: app.resultado_label.config(fg=ACCENT))
    app.after(0, lambda: app.candidato_var.set(""))
    app.after(0, lambda: app.detalle_var.set(""))
    app._set_overlay((255, 184, 48), "Analizando...")
    threading.Thread(target=lambda: _verificar(app), daemon=True).start()


def _verificar(app):
    vectores = []
    intentos = 0
    ultimo   = -1

    while len(vectores) < 8 and intentos < 120 and app.cam_running:
        intentos += 1
        with app._analisis_lock:
            frame_id = app._analisis["frame_id"]
            v        = app._analisis["vector"]
        if frame_id == ultimo:
            time.sleep(0.04)
            continue
        ultimo = frame_id
        if v is not None:
            vectores.append(v)
        time.sleep(0.04)

    app._set_overlay(None, "")
    app.verificando = False

    if not app.cam_running:
        return

    if not vectores:
        app.after(0, lambda: app.resultado_var.set("Sin rostro"))
        app.after(0, lambda: app.resultado_label.config(fg=WARNING))
        app.after(0, lambda: app.candidato_var.set("Ponte frente a la camara."))
        app.after(0, lambda: _set_sim_bar(app, 0, BORDER))
        app.after(3000, lambda: _lanzar_verificacion(app))
        return

    v_final   = np.mean(vectores, axis=0).astype(np.float32)
    resultado = reconocer_persona(v_final)

    if resultado is None:
        hay_registros = len(cargar_vectores_por_angulo()) > 0
        if hay_registros:
            app.after(0, lambda: app.resultado_var.set("ACCESO DENEGADO"))
            app.after(0, lambda: app.resultado_label.config(fg=DANGER))
            app.after(0, lambda: app.candidato_var.set("Persona no reconocida. No esta registrada."))
            app._set_overlay((255, 59, 92), "Desconocido")
        else:
            app.after(0, lambda: app.resultado_var.set("Sin registros"))
            app.after(0, lambda: app.resultado_label.config(fg=WARNING))
            app.after(0, lambda: app.candidato_var.set("No hay usuarios registrados."))
        app.after(0, lambda: _set_sim_bar(app, 0, BORDER))
        app.after(4000, lambda: _lanzar_verificacion(app))
        return

    sim         = resultado["similitud_pct"]
    color_barra = SUCCESS if resultado["acceso"] else (WARNING if sim >= 55 else DANGER)
    app.after(0, lambda: _set_sim_bar(app, int(sim), color_barra))
    app.after(0, lambda r=resultado: app.detalle_var.set(
        f"Angulo match: {r['angulo']}"))

    if resultado["acceso"]:
        app.after(0, lambda r=resultado: _resultado_ok(app, r))
    else:
        app.after(0, lambda r=resultado: _resultado_negado(app, r))


def _resultado_ok(app, r):
    app._set_overlay((0, 255, 136), r["nombre"])
    app.resultado_var.set("ACCESO PERMITIDO")
    app.resultado_label.config(fg=SUCCESS)
    app.candidato_var.set(
        f"{r['nombre']}\n"
        f"Rol: {r.get('rol','---').upper()}\n"
        f"Similitud: {r['similitud_pct']}%")
    app.after(4000, lambda: _lanzar_verificacion(app))


def _resultado_negado(app, r):
    app._set_overlay((255, 59, 92), "Desconocido")
    app.resultado_var.set("ACCESO DENEGADO")
    app.resultado_label.config(fg=DANGER)
    app.candidato_var.set(
        f"Mas parecido a:\n{r['nombre']}\n"
        f"Similitud: {r['similitud_pct']}%")
    app.after(4000, lambda: _lanzar_verificacion(app))


def _monitor_cara(app):
    """
    Monitor continuo durante la pantalla de acceso.
    Si no se detecta cara por TIMEOUT_SIN_CARA segundos:
      - Resetea la pantalla a estado inicial
      - Fuerza un nuevo ciclo de verificacion
    """
    TIMEOUT_SIN_CARA = 1.5

    while app.cam_running:
        with app._analisis_lock:
            v = app._analisis["vector"]

        if v is not None:
            app._ultima_cara_t = time.time()
        else:
            sin_cara = time.time() - app._ultima_cara_t
            if sin_cara >= TIMEOUT_SIN_CARA and not app.verificando:
                app.after(0, lambda: _resetear_pantalla_acceso(app))

        time.sleep(0.15)


def _resetear_pantalla_acceso(app):
    """Limpia la pantalla de acceso al estado inicial de espera."""
    try:
        app.resultado_var.set("Esperando...")
        app.resultado_label.config(fg=ACCENT)
        app.candidato_var.set("")
        app.detalle_var.set("")
        _set_sim_bar(app, 0, BORDER)
        app._set_overlay(None, "")
    except Exception:
        pass