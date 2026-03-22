"""
Pantalla de registro: formulario + escaneo facial de 4 pasos.
Se invoca desde app_base.py como _show_registro().
"""

import tkinter as tk
import threading
import time
import numpy as np

from vistas.app_base import (BG, PANEL, CARD, ACCENT, SUCCESS, DANGER, WARNING,
                       TEXT, SUBTEXT, BORDER, COLOR_ROL,
                       W, H, PANEL_W, CAM_W,
                       PASOS_REGISTRO, N_PASOS, TIEMPO_ESCANEO,
                       MAX_MUESTRAS_PASO, MUESTRAS_MIN_PASO)
from database import registrar_usuario, guardar_vectores_por_angulo, eliminar_persona, ROLES_VALIDOS


def show_registro(app):
    """Construye y muestra la pantalla de registro."""
    app._clear()
    app.geometry(f"{W}x{H}+0+0")
    app._set_overlay(None, "")

    left = tk.Frame(app, bg=PANEL, width=PANEL_W, height=H)
    left.place(x=0, y=0)
    tk.Frame(left, bg=ACCENT, width=PANEL_W, height=3).place(x=0, y=0)
    tk.Label(left, text="REGISTRO",      font=app.f_title, fg=ACCENT,  bg=PANEL).place(x=18, y=8)
    tk.Label(left, text="Nuevo usuario", font=app.f_sub,   fg=SUBTEXT, bg=PANEL).place(x=18, y=34)
    tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=52)

    app._field(left, "Nombre(s)",        app.nombre_var,  58)
    app._field(left, "Apellido paterno",  app.ap_pat_var, 106)
    app._field(left, "Apellido materno",  app.ap_mat_var, 154)

    tk.Label(left, text="Rol", font=app.f_label, fg=SUBTEXT, bg=PANEL).place(x=18, y=204)
    rf = tk.Frame(left, bg=PANEL)
    rf.place(x=18, y=218, width=284)
    for i, rol in enumerate(ROLES_VALIDOS):
        c = COLOR_ROL.get(rol, ACCENT)
        tk.Radiobutton(rf, text=rol.capitalize(),
                       variable=app.rol_var, value=rol,
                       font=app.f_label, fg=c, bg=PANEL,
                       selectcolor=CARD, activebackground=PANEL,
                       activeforeground=c, cursor="hand2"
                       ).grid(row=0, column=i, padx=8)

    tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=244)

    # ── Indicadores de los 4 pasos ────────────────────────────────────────────
    tk.Label(left, text="PASOS DE ESCANEO:", font=app.f_zona,
             fg=SUBTEXT, bg=PANEL).place(x=18, y=250)

    app._paso_frames = []
    paso_w = 56
    for i, (_, icono, _, etiq, _, _, _, _) in enumerate(PASOS_REGISTRO):
        fx = 18 + i * (paso_w + 6)
        pf = tk.Frame(left, bg=BORDER, width=paso_w, height=52)
        pf.place(x=fx, y=262)
        li = tk.Label(pf, text=str(i+1), font=app.f_btn,  fg=SUBTEXT, bg=BORDER)
        li.place(relx=.5, y=10, anchor="center")
        ln = tk.Label(pf, text=icono,    font=app.f_label, fg=SUBTEXT, bg=BORDER)
        ln.place(relx=.5, y=28, anchor="center")
        bpf = tk.Frame(pf, bg="#111", width=paso_w-4, height=4)
        bpf.place(x=2, y=46)
        bp = tk.Frame(bpf, bg=SUBTEXT, width=0, height=4)
        bp.place(x=0, y=0)
        app._paso_frames.append((pf, li, ln, bp))

    app.paso_desc_var = tk.StringVar(value="Completa los campos e inicia")
    tk.Label(left, textvariable=app.paso_desc_var,
             font=app.f_label, fg=ACCENT, bg=PANEL,
             wraplength=284, justify="center"
             ).place(x=18, y=322, width=284)

    app.cap_btn = tk.Button(
        left, text="INICIAR ESCANEO", font=app.f_btn,
        fg=BG, bg=ACCENT, relief="flat", cursor="hand2",
        padx=8, pady=7, command=lambda: _iniciar_registro(app))
    app.cap_btn.place(x=18, y=348, width=284)
    app.cap_btn.bind("<Enter>", lambda e: app.cap_btn.config(bg=app._lighten(ACCENT)))
    app.cap_btn.bind("<Leave>", lambda e: app.cap_btn.config(bg=ACCENT))

    tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=390)

    tk.Label(left, text="MUESTRAS POR PASO:", font=app.f_zona,
             fg=SUBTEXT, bg=PANEL).place(x=18, y=396)

    app._barra_pasos = []
    etiquetas_pasos  = ["Frente", "Derecha", "Izquierda", "Frente"]
    barra_w = 56
    for i, etiq in enumerate(etiquetas_pasos):
        fx = 18 + i * (barra_w + 6)
        tk.Label(left, text=etiq[:5], font=app.f_zona,
                 fg=SUBTEXT, bg=PANEL).place(x=fx, y=410)
        bg_b = tk.Frame(left, bg=BORDER, width=barra_w, height=8)
        bg_b.place(x=fx, y=422)
        bar  = tk.Frame(bg_b, bg=BORDER, width=0, height=8)
        bar.place(x=0, y=0)
        lbl  = tk.Label(left, text="0", font=app.f_zona, fg=SUBTEXT, bg=PANEL)
        lbl.place(x=fx + barra_w//2, y=432, anchor="center")
        app._barra_pasos.append((bar, lbl, barra_w))

    tk.Frame(left, bg=BORDER, height=1, width=284).place(x=18, y=446)

    app.progreso_var = tk.StringVar(value="")
    app.prog_label   = tk.Label(left, textvariable=app.progreso_var,
                                 font=app.f_status, fg=WARNING, bg=PANEL,
                                 justify="center", wraplength=284)
    app.prog_label.place(x=18, y=454, width=284)

    app.timer_var    = tk.StringVar(value="")
    app.paso_txt_var = tk.StringVar(value="")
    tk.Label(left, textvariable=app.timer_var,
             font=app.f_title, fg=ACCENT, bg=PANEL
             ).place(x=18, y=500, anchor="nw")
    tk.Label(left, textvariable=app.paso_txt_var,
             font=app.f_zona, fg=SUBTEXT, bg=PANEL,
             wraplength=160).place(x=62, y=508, anchor="nw")

    tk.Button(left, text="<  Volver", font=app.f_label,
              fg=SUBTEXT, bg=PANEL, relief="flat",
              cursor="hand2", command=app._volver).place(x=18, y=565)

    right = tk.Frame(app, bg=BG, width=CAM_W, height=H)
    right.place(x=PANEL_W, y=0)
    app.cam_label = tk.Label(right, bg="#080A0F")
    app.cam_label.place(x=8, y=5, width=CAM_W-16, height=H-46)

    tk.Label(right, textvariable=app.status_var,
             font=app.f_status, fg=ACCENT, bg=BG).place(x=8, y=H-36)

    app.prog_frame = tk.Frame(right, bg=BORDER, width=CAM_W-16, height=8)
    app.prog_frame.place(x=8, y=H-18)
    app.prog_bar = tk.Frame(app.prog_frame, bg=ACCENT, width=0, height=8)
    app.prog_bar.place(x=0, y=0)

    app._start_cam()
    app.cam_running = True
    threading.Thread(target=app._loop_camara,
                     kwargs={"max_w": CAM_W-16, "max_h": H-46},
                     daemon=True).start()
    threading.Thread(target=app._loop_analisis, daemon=True).start()


# ── Lógica de registro ────────────────────────────────────────────────────────

def _iniciar_registro(app):
    nombre = app.nombre_var.get().strip()
    ap_pat = app.ap_pat_var.get().strip()
    ap_mat = app.ap_mat_var.get().strip()
    rol    = app.rol_var.get()

    if not nombre or not ap_pat:
        app.progreso_var.set("Nombre y apellido paterno son obligatorios.")
        app.prog_label.config(fg=DANGER)
        return

    app.cap_btn.config(state="disabled", bg=BORDER, text="Escaneando...")
    app.progreso_var.set("Preparando...")
    app.prog_label.config(fg=WARNING)
    app.timer_var.set("")
    app.after(0, _resetear_pasos_ui, app)

    threading.Thread(
        target=_capturar_registro,
        args=(app, nombre, ap_pat, ap_mat, rol),
        daemon=True).start()


def _capturar_registro(app, nombre, ap_pat, ap_mat, rol):
    BAR_W = CAM_W - 16

    uid = registrar_usuario(
        nombre=nombre,
        apellido_paterno=ap_pat,
        apellido_materno=ap_mat,
        rol=rol)

    if uid == -1:
        app.after(0, lambda: app._safe(
            lambda: app.progreso_var.set("Error al registrar. Intentalo de nuevo.")))
        app.after(0, lambda: app._safe(lambda: app.prog_label.config(fg=DANGER)))
        app.after(0, lambda: app._safe(
            lambda: app.cap_btn.config(state="normal", bg=ACCENT,
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

        app._modo_deteccion = modo_det
        app._tipo_esperado  = tipo_esperado

        app.after(0, lambda i=instruccion: app._safe(
            lambda: app.status_var.set(i)))
        app.after(0, lambda pi=paso_idx: _activar_paso_ui(app, pi, 0.0))
        app._set_overlay((255, 184, 48), f"{paso_idx+1}/{N_PASOS}: {instruccion}")

        while t_paso_activo < duracion and app.cam_running:

            if len(vectores_paso) >= MAX_MUESTRAS_PASO:
                t_paso_activo = duracion
                break

            with app._analisis_lock:
                frame_id = app._analisis["frame_id"]
                v        = app._analisis["vector"]
                tipo_det = app._analisis["tipo"]

            angulo_ok     = (tipo_det == tipo_esperado) and (v is not None)
            cara_presente = (v is not None)

            ahora = time.time()
            if angulo_ok:
                if t_ultimo_tick is not None:
                    t_paso_activo += ahora - t_ultimo_tick
                t_ultimo_tick = ahora
            else:
                t_ultimo_tick = None

            restante      = max(0.0, duracion - t_paso_activo)
            progreso_paso = min(1.0, t_paso_activo / duracion)
            elapsed_g     = t_offsets[paso_idx] + t_paso_activo
            progreso_tot  = min(1.0, elapsed_g / TIEMPO_ESCANEO)

            app.after(0, lambda s=int(restante)+1: app._safe(
                lambda: app.timer_var.set(f"{s}s")))
            app.after(0, lambda pt=int(progreso_tot*BAR_W): app._safe(
                lambda: app.prog_bar.config(width=pt)))
            app.after(0, lambda pi=paso_idx, pp=progreso_paso:
                       _activar_paso_ui(app, pi, pp))

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
                    app._set_overlay(
                        (0, 255, 136),
                        f"{paso_idx+1}/{N_PASOS} {etiqueta} [{n}/{MAX_MUESTRAS_PASO}]")
                    app.after(0, lambda nt=tot: app._safe(
                        lambda: app.progreso_var.set(f"Total: {nt} muestras")))
                    app.after(0, lambda: app._safe(
                        lambda: app.prog_label.config(fg=SUCCESS)))
                    app.after(0, lambda pi=paso_idx, nm=n_muestras_paso[paso_idx]:
                               _update_barra_paso(app, pi, nm))

                elif cara_presente and not angulo_ok:
                    app._set_overlay((255, 59, 92),
                                      f"ESPERANDO — {msg_correccion}")
                    app.after(0, lambda mc=msg_correccion: app._safe(
                        lambda: app.status_var.set(mc)))
                    app.after(0, lambda: app._safe(
                        lambda: app.prog_label.config(fg=DANGER)))
                else:
                    app._set_overlay((255, 184, 48),
                                      f"{paso_idx+1}/{N_PASOS}: {instruccion}")
                    app.after(0, lambda i=instruccion: app._safe(
                        lambda: app.status_var.set(i)))
                    app.after(0, lambda: app._safe(
                        lambda: app.prog_label.config(fg=WARNING)))

            time.sleep(0.04)

        app.after(0, lambda pi=paso_idx: _activar_paso_ui(app, pi+1, 0.0))

    # ── Fin escaneo ───────────────────────────────────────────────────────────
    app._modo_deteccion = "auto"
    app._tipo_esperado  = None
    app._set_overlay(None, "")
    app.after(0, lambda: app._safe(lambda: app.timer_var.set("")))
    app.after(0, lambda: app._safe(lambda: app.paso_txt_var.set("")))
    app.after(0, lambda: app._safe(lambda: app.prog_bar.config(width=BAR_W)))

    total_muestras  = sum(n_muestras_paso)
    pasos_ok        = sum(1 for n in n_muestras_paso if n >= MUESTRAS_MIN_PASO)
    nombre_completo = f"{nombre} {ap_pat} {ap_mat}".strip()

    if pasos_ok == N_PASOS:
        guardar_vectores_por_angulo(uid, vectores_angulo)
        app.after(0, lambda: app._safe(lambda: app.prog_bar.config(bg=SUCCESS)))
        app.after(0, lambda: app._safe(lambda: app.status_var.set(
            f"Registro completo — {total_muestras} muestras, {pasos_ok} pasos")))
        app.after(0, lambda nv=nombre_completo, nt=total_muestras: app._safe(
            lambda: app.progreso_var.set(
                f"Listo. {nv}\n{nt} muestras en {N_PASOS} pasos.")))
        app.after(0, lambda: app._safe(lambda: app.prog_label.config(fg=SUCCESS)))
        app.after(0, lambda: app._safe(lambda: app.cap_btn.config(
            state="normal", bg=SUCCESS, text="REGISTRO COMPLETO")))
        for var in (app.nombre_var, app.ap_pat_var, app.ap_mat_var):
            app.after(0, lambda v=var: app._safe(lambda: v.set("")))
        app.after(0, lambda: app._safe(lambda: app.rol_var.set("estudiante")))
        app.after(3500, lambda: app._safe(lambda: app.cap_btn.config(
            bg=ACCENT, text="INICIAR ESCANEO")))
    else:
        try:
            eliminar_persona(uid)
        except:
            pass
        pasos_fallidos = [PASOS_REGISTRO[i][3]
                          for i, n in enumerate(n_muestras_paso)
                          if n < MUESTRAS_MIN_PASO]
        app.after(0, lambda pf=pasos_fallidos: app._safe(
            lambda: app.progreso_var.set(
                f"Pasos incompletos:\n{', '.join(pf)}\n"
                f"Intentalo de nuevo.")))
        app.after(0, lambda: app._safe(lambda: app.prog_label.config(fg=DANGER)))
        app.after(0, lambda: app._safe(lambda: app.status_var.set(
            "Intentalo de nuevo. Acercate mas a la camara.")))
        app.after(0, lambda: app._safe(lambda: app.cap_btn.config(
            state="normal", bg=ACCENT, text="INICIAR ESCANEO")))
        app.after(0, lambda: _resetear_pasos_ui(app))


# ── UI helpers ────────────────────────────────────────────────────────────────

def _activar_paso_ui(app, paso_idx, progreso=0.0):
    if not hasattr(app, "_paso_frames") or not app.cam_running:
        return
    paso_w = 56
    try:
        for i, (pf, li, ln, bp) in enumerate(app._paso_frames):
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


def _resetear_pasos_ui(app):
    if not hasattr(app, "_paso_frames"):
        return
    try:
        for pf, li, ln, bp in app._paso_frames:
            pf.config(bg=BORDER); li.config(fg=SUBTEXT, bg=BORDER)
            ln.config(fg=SUBTEXT, bg=BORDER); bp.master.config(bg="#111")
            bp.config(width=0)
    except:
        pass


def _update_barra_paso(app, paso_idx, n_muestras):
    if not hasattr(app, "_barra_pasos") or paso_idx >= len(app._barra_pasos):
        return
    try:
        bar, lbl, barra_w = app._barra_pasos[paso_idx]
        pct = min(1.0, n_muestras / MAX_MUESTRAS_PASO)
        w   = int(pct * barra_w)
        col = SUCCESS if pct >= 1.0 else ACCENT if pct > 0.4 else WARNING
        bar.config(width=w, bg=col)
        lbl.config(text=str(n_muestras), fg=col)
    except:
        pass