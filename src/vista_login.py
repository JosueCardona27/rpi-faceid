"""
vista_login.py
==============
Pantalla de login institucional + menú principal post-login.
Modo luz · inputs redondeados · tarjeta redondeada · botón redondeado.
"""

import tkinter as tk

# ── Paleta ────────────────────────────────────────────────────────────────────
BG      = "#F5F0E8"
PANEL   = "#FFFFFF"
CARD    = "#EAE5D8"
ACCENT  = "#2D6A4F"
ACCENT2 = "#40916C"
SUCCESS = "#52B788"
DANGER  = "#C1121F"
WARNING = "#E07A00"
TEXT    = "#1A1A2E"
SUBTEXT = "#5C6170"
BORDER  = "#C8C2B2"
NAVY    = "#1B2A4A"
TEAL    = "#0A8F70"
GOLD    = "#B5860D"

W, H = 1024, 600


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS DE DIBUJO
# ════════════════════════════════════════════════════════════════════════════

def _round_rect(cv, x1, y1, x2, y2, r=16, fill=PANEL, outline=BORDER, width=1, tags=""):
    """Dibuja un rectángulo con esquinas redondeadas en un Canvas."""
    cv.create_arc(x1,     y1,     x1+2*r, y1+2*r, start=90,  extent=90, fill=fill, outline=fill, tags=tags)
    cv.create_arc(x2-2*r, y1,     x2,     y1+2*r, start=0,   extent=90, fill=fill, outline=fill, tags=tags)
    cv.create_arc(x1,     y2-2*r, x1+2*r, y2,     start=180, extent=90, fill=fill, outline=fill, tags=tags)
    cv.create_arc(x2-2*r, y2-2*r, x2,     y2,     start=270, extent=90, fill=fill, outline=fill, tags=tags)
    cv.create_rectangle(x1+r, y1,   x2-r, y2,     fill=fill, outline="",      tags=tags)
    cv.create_rectangle(x1,   y1+r, x2,   y2-r,   fill=fill, outline="",      tags=tags)
    if width > 0:
        cv.create_arc(x1,     y1,     x1+2*r, y1+2*r, start=90,  extent=90, style="arc", outline=outline, width=width, tags=tags)
        cv.create_arc(x2-2*r, y1,     x2,     y1+2*r, start=0,   extent=90, style="arc", outline=outline, width=width, tags=tags)
        cv.create_arc(x1,     y2-2*r, x1+2*r, y2,     start=180, extent=90, style="arc", outline=outline, width=width, tags=tags)
        cv.create_arc(x2-2*r, y2-2*r, x2,     y2,     start=270, extent=90, style="arc", outline=outline, width=width, tags=tags)
        cv.create_line(x1+r, y1,   x2-r, y1,   fill=outline, width=width, tags=tags)
        cv.create_line(x1+r, y2,   x2-r, y2,   fill=outline, width=width, tags=tags)
        cv.create_line(x1,   y1+r, x1,   y2-r, fill=outline, width=width, tags=tags)
        cv.create_line(x2,   y1+r, x2,   y2-r, fill=outline, width=width, tags=tags)


def _rounded_entry(parent, var, show=None, width=320, height=40):
    """
    Input redondeado real:
    - Canvas dibuja el fondo + borde redondeado con _round_rect
    - Entry encima sin bordes
    - Al enfocar, redibuja el borde en ACCENT (verde)
    """
    r = 12
    frame = tk.Frame(parent, bg=PANEL, width=width, height=height)
    frame.pack_propagate(False)

    cv = tk.Canvas(frame, width=width, height=height,
                   bg=PANEL, highlightthickness=0)
    cv.place(x=0, y=0)

    def _draw(border_col):
        cv.delete("all")
        _round_rect(cv, 1, 1, width-1, height-1,
                    r=r, fill=CARD, outline=border_col, width=2)

    _draw(BORDER)

    kwargs = dict(
        textvariable=var,
        font=("Helvetica", 11),
        fg=TEXT, bg=CARD,
        insertbackground=ACCENT,
        relief="flat", highlightthickness=0, bd=0
    )
    if show:
        kwargs["show"] = show

    entry = tk.Entry(frame, **kwargs)
    entry.place(x=r+8, y=6, width=width - 2*r - 16, height=height - 12)

    entry.bind("<FocusIn>",  lambda e: _draw(ACCENT))
    entry.bind("<FocusOut>", lambda e: _draw(BORDER))

    return frame, entry


def _rounded_button(parent, text, cmd, width=320, height=46,
                    bg=NAVY, fg="#FFFFFF", hover=ACCENT2, font_size=12):
    """
    Botón redondeado: Canvas con _round_rect + Label clicable encima.
    Tkinter no soporta border-radius en Button nativo, así que se simula.
    """
    r = 14
    frame = tk.Frame(parent, bg=PANEL, width=width, height=height)
    frame.pack_propagate(False)

    cv = tk.Canvas(frame, width=width, height=height,
                   bg=PANEL, highlightthickness=0)
    cv.place(x=0, y=0)

    def _draw(color):
        cv.delete("all")
        _round_rect(cv, 0, 0, width, height, r=r, fill=color, outline=color, width=0)

    _draw(bg)

    lbl = tk.Label(frame, text=text,
                   font=("Helvetica", font_size, "bold"),
                   fg=fg, bg=bg, cursor="hand2")
    lbl.place(relx=.5, rely=.5, anchor="center")

    def on_enter(e):
        _draw(hover)
        lbl.config(bg=hover)

    def on_leave(e):
        _draw(bg)
        lbl.config(bg=bg)

    def on_click(e):
        cmd()

    for w in (cv, lbl):
        w.bind("<Enter>",   on_enter)
        w.bind("<Leave>",   on_leave)
        w.bind("<Button-1>", on_click)

    return frame


#  PANTALLA LOGIN

def build_main(app):
    app._clear()
    app.geometry(f"{W}x{H}+0+0")
    app.configure(bg=BG)

    cv = tk.Canvas(app, width=W, height=H, bg=BG, highlightthickness=0)
    cv.place(x=0, y=0)

    # ── barra superior ────────────────────────────────────────────────────────
    cv.create_rectangle(0, 0, W, 50, fill=NAVY, outline="")
    cv.create_text(W//2, 25,
                   text="SISTEMA DE CONTROL DE ACCESO",
                   font=("Helvetica", 13, "bold"), fill="#FFFFFF")
    cv.create_rectangle(0, 50, W, 53, fill=GOLD,   outline="")
    cv.create_rectangle(0, 53, W, 56, fill=ACCENT, outline="")

    # ── barra inferior ────────────────────────────────────────────────────────
    cv.create_rectangle(0, H-28, W, H, fill=NAVY, outline="")
    cv.create_text(W//2, H-14,
                   text="Universidad de Colima",
                   font=("Helvetica", 8), fill="#AABBCC")

    # ── lado izquierdo ────────────────────────────────────────────────────────
    LEFT_W = 490
    cv.create_rectangle(0, 56, LEFT_W, H-28, fill="#EDE8DC", outline="")
    cv.create_rectangle(LEFT_W, 56, LEFT_W+1, H-28, fill=BORDER, outline="")

    cv.create_text(LEFT_W//2, 94,
                   text="Facultad de Ingeniería",
                   font=("Helvetica", 16, "bold"), fill=NAVY)
    cv.create_text(LEFT_W//2, 118,
                   text="Electromecanica",
                   font=("Helvetica", 13), fill=ACCENT2)
    cv.create_rectangle(LEFT_W//2-90, 132, LEFT_W//2+90, 134,
                        fill=ACCENT, outline="")

    _draw_mascota(cv, LEFT_W//2, 310)

    # ── lado derecho blanco ───────────────────────────────────────────────────
    RX = LEFT_W + 1
    RW = W - RX
    cv.create_rectangle(RX, 56, W, H-28, fill=PANEL, outline="")

    cv.create_text(RX + RW//2, 96,
                   text="Iniciar Sesión",
                   font=("Helvetica", 20, "bold"), fill=NAVY)
    cv.create_text(RX + RW//2, 120,
                   text="Ingresa tus credenciales institucionales",
                   font=("Helvetica", 9), fill=SUBTEXT)
    cv.create_rectangle(RX+32, 133, W-32, 135, fill=BORDER, outline="")

    # tarjeta del formulario
    FX = RX + 28
    FY = 146
    FW = RW - 56
    FH = 310
# 1. sombra PRIMERO
    _round_rect(cv, FX+4, FY+4, FX+FW+4, FY+FH+4,
                r=16, fill="#D8D2C2", outline="#D8D2C2", width=0)

    # 2. tarjeta — _round_rect completo, luego tapa esquinas superiores con cuadrado
    _round_rect(cv, FX, FY, FX+FW, FY+FH,
                r=16, fill=PANEL, outline=BORDER, width=1)
    # tapar esquinas redondeadas de arriba con cuadrados blancos
    cv.create_rectangle(FX, FY, FX+16, FY+16, fill=PANEL, outline="")
    cv.create_rectangle(FX+FW-16, FY, FX+FW, FY+16, fill=PANEL, outline="")
    # redibujar borde recto superior y laterales superiores
    cv.create_line(FX,    FY,    FX+FW, FY,    fill=BORDER, width=1)
    cv.create_line(FX,    FY,    FX,    FY+16, fill=BORDER, width=1)
    cv.create_line(FX+FW, FY,    FX+FW, FY+16, fill=BORDER, width=1)

    # 3. franja tricolor
    cv.create_rectangle(FX, FY,   FX+FW,    FY+3, fill=NAVY,   outline="")
    cv.create_rectangle(FX, FY+3, FX+FW//2, FY+6, fill=ACCENT, outline="")
    cv.create_rectangle(FX+FW//2, FY+3, FX+FW, FY+6, fill=TEAL, outline="")
    # ── variables ─────────────────────────────────────────────────────────────
    app.login_cuenta_var = tk.StringVar()
    app.login_pass_var   = tk.StringVar()
    app.login_msg_var    = tk.StringVar(value="")

    EW = FW - 44

    # campo cuenta
    lbl_y1 = FY + 28
    cv.create_text(FX+22, lbl_y1,
                   text="Número de Cuenta", anchor="w",
                   font=("Helvetica", 10, "bold"), fill=TEXT)

    entry_y1 = lbl_y1 + 22
    frame1, entry_cuenta = _rounded_entry(app, app.login_cuenta_var,
                                          width=EW, height=40)
    frame1.place(x=FX+22, y=entry_y1)

    # campo contraseña
    lbl_y2 = entry_y1 + 62
    cv.create_text(FX+22, lbl_y2,
                   text="Contraseña", anchor="w",
                   font=("Helvetica", 10, "bold"), fill=TEXT)

    entry_y2 = lbl_y2 + 22
    frame2, entry_pass = _rounded_entry(app, app.login_pass_var,
                                        show="●", width=EW, height=40)
    frame2.place(x=FX+22, y=entry_y2)

    # mensaje error
    app._login_msg_label = tk.Label(
        app, textvariable=app.login_msg_var,
        font=("Helvetica", 9), fg=DANGER, bg=PANEL,
        wraplength=EW)
    app._login_msg_label.place(x=FX+22, y=entry_y2+48, width=EW)

    # botón redondeado
    btn_y = FY + FH - 68
    btn_frame = _rounded_button(
        app, text="ACCEDER",
        cmd=lambda: _intentar_login(app),
        width=EW, height=46,
        bg=NAVY, fg="#FFFFFF", hover=ACCENT2)
    btn_frame.place(x=FX+22, y=btn_y)

    # binds teclado
    entry_cuenta.bind("<Return>", lambda e: entry_pass.focus_set())
    entry_pass.bind("<Return>",   lambda e: _intentar_login(app))
    entry_cuenta.focus_set()


def _draw_mascota(cv, cx, cy):
    R = 115
    cv.create_oval(cx-R+5, cy-R+5, cx+R+5, cy+R+5, fill="#D0C9B8", outline="")
    cv.create_oval(cx-R, cy-R, cx+R, cy+R, fill="#E8E2D4", outline=ACCENT2, width=2)
    cv.create_rectangle(cx-50, cy-40, cx+50, cy+30, outline=BORDER, width=2, dash=(4,4))
    cv.create_polygon(cx-35, cy+25, cx-8, cy-8, cx+10, cy+12,
                      cx+24, cy-6, cx+44, cy+25, fill=BORDER, outline="")
    cv.create_oval(cx-34, cy-34, cx-18, cy-18, fill=GOLD, outline="")
    cv.create_text(cx, cy+60, text="[ Logo / Mascota ]",
                   font=("Helvetica", 9, "italic"), fill=SUBTEXT)


# ── lógica login ──────────────────────────────────────────────────────────────

def _intentar_login(app):
    # ── MODO PRUEBA: entra directo sin validar nada ───────────────────────────
    app._build_main()
    return
    # ── descomentar cuando la BD esté lista ──────────────────────────────────
    # cuenta = app.login_cuenta_var.get().strip()
    # passwd = app.login_pass_var.get().strip()
    # if not cuenta or not passwd:
    #     app.login_msg_var.set("⚠  Completa todos los campos.")
    #     app._login_msg_label.config(fg=WARNING)
    #     return
    # app.login_msg_var.set("")
    # app._build_main()

    # PRODUCCIÓN (descomentar cuando esté lista la BD):
    # try:
    #     from database import verificar_credenciales_admin
    #     rol = verificar_credenciales_admin(cuenta, passwd)
    # except Exception as ex:
    #     app.login_msg_var.set(f"Error: {ex}")
    #     app._login_msg_label.config(fg=DANGER)
    #     return
    # if rol is None:
    #     app.login_msg_var.set("✗  Credenciales incorrectas.")
    #     app._login_msg_label.config(fg=DANGER)
    #     app.login_pass_var.set("")
    # else:
    #     app.login_msg_var.set("")
    #     app._build_main()   # llama al menú principal de interfaz.py


# fin de vista_login.py