"""
Pantalla principal: dos botones (REGISTRARME / ACCESO).
Se invoca desde app_base.py como _build_main().
"""

import tkinter as tk
from vistas.app_base import (BG, PANEL, CARD, ACCENT, ACCENT2, SUCCESS,
                       DANGER, WARNING, TEXT, SUBTEXT, BORDER,
                       W, H, PANEL_W, CAM_W,
                       N_PASOS, TIEMPO_ESCANEO, MAX_MUESTRAS_PASO, USAR_PICAM)


def build_main(app):
    """Construye y muestra la pantalla principal."""
    app._clear()
    app.geometry(f"{W}x{H}+0+0")

    cv = tk.Canvas(app, width=W, height=H, bg=BG, highlightthickness=0)
    cv.place(x=0, y=0)
    cv.create_rectangle(0, 0, W, 3, fill=ACCENT, outline="")

    # Icono de cara
    IY = 30; cx = W // 2
    cv.create_oval(cx-20, IY,    cx+20, IY+44, outline=ACCENT,  width=2)
    cv.create_oval(cx-11, IY+9,  cx+11, IY+33, outline=ACCENT2, width=2)
    cv.create_line(cx,    IY+44, cx,    IY+62, fill=ACCENT, width=2)
    cv.create_line(cx-20, IY+53, cx+20, IY+53, fill=ACCENT, width=2)

    TY = IY + 74
    tk.Label(app, text="SISTEMA DE ACCESO FACIAL",
             font=app.f_title, fg=ACCENT, bg=BG
             ).place(x=W//2, y=TY, anchor="center")

    SY = TY + 22
    modo_txt = "Raspberry Pi" if USAR_PICAM else "Webcam"
    tk.Label(app,
             text=f"LBP 512 dims  |  4 pasos  |  {modo_txt}",
             font=app.f_sub, fg=SUBTEXT, bg=BG
             ).place(x=W//2, y=SY, anchor="center")

    LY = SY + 16
    cv.create_line(W//2-280, LY, W//2+280, LY, fill=BORDER, width=1)

    CY = LY + 16
    CW, CH = 230, 180; GAP = 40
    CX1 = W//2 - (CW*2+GAP)//2
    CX2 = CX1 + CW + GAP

    app._card_btn(cv, CX1, CY, CW, CH, "REGISTRARME",
                  "Registrar nuevo usuario\ny capturar datos faciales",
                  ACCENT, app._show_registro)
    app._card_btn(cv, CX2, CY, CW, CH, "ACCESO",
                  "Verificar identidad\nmediante reconocimiento facial",
                  SUCCESS, app._show_acceso)

    cv.create_line(0, H-26, W, H-26, fill=BORDER, width=1)
    tk.Label(app,
             text=f"v5.5  |  4 pasos ({int(TIEMPO_ESCANEO)}s)  |  "
                  f"max {MAX_MUESTRAS_PASO} muestras/paso  |  {modo_txt}",
             font=app.f_zona, fg=SUBTEXT, bg=BG
             ).place(x=W//2, y=H-13, anchor="center")