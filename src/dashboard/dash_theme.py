"""
dash_theme.py
=============
Paleta de colores y utilidades de UI compartidas.

Columnas en make_treeview aceptan 3 ó 4 elementos:
    (id, cabecera, ancho)              → anchor "w" por defecto
    (id, cabecera, ancho, anchor)      → anchor personalizado ("w","center","e")

CAMBIOS:
  · scrollable_frame usa bind en Enter/Leave en lugar de bind_all,
    evitando que dos áreas scrollables (nav + contenido) interfieran.
"""
import tkinter as tk
from tkinter import ttk
from typing import Tuple, Union, Optional

# ── Colores ─────────────────────────────────────────────────────
BG      = "#0A0E1A"
SIDEBAR = "#151923"
CARD    = "#1A1F2E"
CARD2   = "#232937"
ACCENT  = "#00D4AA"
BLUE    = "#4A9EFF"
AMBER   = "#FBBF24"
RED     = "#F87171"
PURPLE  = "#C084FC"
GREEN   = "#34D399"
T1      = "#F8FAFC"
T2      = "#CBD5E1"
T3      = "#64748B"
BORDER  = "#334155"
SHADOW  = "#00000015"


# ── Estilos ttk ─────────────────────────────────────────────────
def aplicar_estilo_treeview():
    style = ttk.Style()
    try:
        style.theme_use("default")
    except Exception:
        pass
    style.configure(
        "Dark.Treeview",
        background=CARD, foreground=T1,
        fieldbackground=CARD,
        rowheight=30, font=("Arial", 9), borderwidth=0,
    )
    style.configure(
        "Dark.Treeview.Heading",
        background=SIDEBAR, foreground=T2,
        font=("Arial", 9, "bold"), relief="flat",
    )
    style.map(
        "Dark.Treeview",
        background=[("selected", "#2D4A6B")],
        foreground=[("selected", T1)],
    )


# ── Helpers ─────────────────────────────────────────────────────
def rounded_card(parent, bg_color=CARD, border_color=BORDER, radius=8) -> tk.Frame:
    """Crea un frame con efecto de borde redondeado simulado."""
    frame = tk.Frame(parent, bg=border_color, relief="flat")
    inner = tk.Frame(frame, bg=bg_color, relief="flat")
    inner.pack(padx=1, pady=1, fill="both", expand=True)
    frame.inner = inner
    return frame

def modern_button(parent, text, command=None, bg_color=ACCENT, fg_color="white",
                 hover_color=None, font_size=10, padding=(12, 8)):
    """Crea un botón moderno con hover effects."""
    if hover_color is None:
        hover_color = adjust_brightness(bg_color, 20)

    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg_color, fg=fg_color,
        activebackground=hover_color, activeforeground=fg_color,
        relief="flat", cursor="hand2",
        font=("Segoe UI", font_size, "normal"),
        padx=padding[0], pady=padding[1],
        borderwidth=0
    )
    return btn

def adjust_brightness(hex_color, percent):
    """Ajusta el brillo de un color hex."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    new_rgb = tuple(min(255, max(0, int(c * (1 + percent/100)))) for c in rgb)
    return '#{:02x}{:02x}{:02x}'.format(*new_rgb)

def card_head(parent, title: str, subtitle: str = "",
             compact: bool = False) -> tk.Frame:
    """
    Cabecera de tarjeta.
    compact=True → título y subtítulo apilados verticalmente (evita desborde en 600 px).
    compact=False → título y subtítulo en la misma fila (comportamiento original).
    """
    h = tk.Frame(parent, bg=parent["bg"])
    h.pack(fill="x", pady=(0, 8 if compact else 12))
    if compact:
        tk.Label(h, text=title, bg=parent["bg"], fg=T1,
                 font=("Segoe UI", 11, "bold"),
                 anchor="w").pack(fill="x")
        if subtitle:
            tk.Label(h, text=subtitle, bg=parent["bg"], fg=T3,
                     font=("Segoe UI", 7),
                     anchor="w").pack(fill="x")
    else:
        tk.Label(h, text=title, bg=parent["bg"], fg=T1,
                 font=("Segoe UI", 12, "bold")).pack(side="left")
        if subtitle:
            tk.Label(h, text=subtitle, bg=parent["bg"], fg=T3,
                     font=("Segoe UI", 9)).pack(side="left", padx=10)
    return h


def scrollable_frame(parent) -> Tuple[tk.Canvas, tk.Frame]:
    """
    Área scrollable. Devuelve (canvas, inner_frame).

    El scroll por rueda del ratón se activa sólo cuando el puntero está
    dentro del canvas/frame, evitando conflictos entre el nav-canvas del
    sidebar y el canvas de contenido.
    """
    canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
    sb     = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    inner  = tk.Frame(canvas, bg=BG)

    def configure_scroll_region(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    inner.bind("<Configure>", configure_scroll_region)

    wid = canvas.create_window((0, 0), window=inner, anchor="nw")

    def configure_inner_width(event=None):
        canvas.itemconfig(wid, width=event.width)
    canvas.bind("<Configure>", configure_inner_width)

    canvas.configure(yscrollcommand=sb.set)
    canvas.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    # ── Scroll con rueda: activo sólo cuando el ratón está dentro ──
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_linux_scroll(event):
        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")

    def _bind_scroll(event=None):
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>",   _on_linux_scroll)
        canvas.bind_all("<Button-5>",   _on_linux_scroll)

    def _unbind_scroll(event=None):
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    canvas.bind("<Enter>", _bind_scroll)
    canvas.bind("<Leave>", _unbind_scroll)
    inner.bind("<Enter>",  _bind_scroll)
    inner.bind("<Leave>",  _unbind_scroll)

    return canvas, inner


def make_treeview(parent, columns: list[tuple], height: int = 15,
                  xscroll: bool = False) -> ttk.Treeview:
    """
    Crea un Treeview oscuro con scrollbar vertical y, opcionalmente, horizontal.

    columns : lista de tuplas (id, cabecera, ancho) o (id, cabecera, ancho, anchor).
    xscroll : True para añadir barra de scroll horizontal (útil en pantallas angostas).
    anchor por defecto: "center".
    """
    col_ids = [c[0] for c in columns]
    tree    = ttk.Treeview(parent, columns=col_ids, show="headings",
                            style="Dark.Treeview", height=height)

    for col_def in columns:
        col_id  = col_def[0]
        heading = col_def[1]
        width   = col_def[2]
        anchor  = col_def[3] if len(col_def) > 3 else "center"
        tree.heading(col_id, text=heading, anchor=anchor)
        # stretch=False para que el scroll horizontal funcione correctamente
        tree.column(col_id, width=width, minwidth=width, stretch=False, anchor=anchor)

    vsb = ttk.Scrollbar(parent, orient="vertical",   command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)

    if xscroll:
        xsb = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=xsb.set)
        # Orden de empaque importante: bottom antes de left/right
        xsb.pack(side="bottom", fill="x")
        vsb.pack(side="right",  fill="y")
        tree.pack(side="left",  fill="both", expand=True)
    else:
        tree.column(col_ids[0], stretch=True)  # sin xscroll, el último se expande
        vsb.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)

    return tree


def minutos_a_texto(mins: int) -> str:
    if mins < 60:
        return f"{mins} min"
    h, m = divmod(mins, 60)
    return f"{h}h {m}min" if m else f"{h}h"


def iniciales(nombre: str, apellido: str = "") -> str:
    n = (nombre or " ").strip()
    a = (apellido or " ").strip()
    return ((n[0] if n else "?") + (a[0] if a else "?")).upper()