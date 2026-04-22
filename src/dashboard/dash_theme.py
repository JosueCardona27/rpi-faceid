"""
dash_theme.py
=============
Paleta de colores y utilidades de UI compartidas.

Columnas en make_treeview aceptan 3 ó 4 elementos:
    (id, cabecera, ancho)              → anchor "w" por defecto
    (id, cabecera, ancho, anchor)      → anchor personalizado ("w","center","e")
"""
import tkinter as tk
from tkinter import ttk

# ── Colores ─────────────────────────────────────────────────────
BG      = "#0F1117"
SIDEBAR = "#161B26"
CARD    = "#1E2638"
CARD2   = "#252E42"
ACCENT  = "#00C9A7"
BLUE    = "#3B82F6"
AMBER   = "#F59E0B"
RED     = "#EF4444"
PURPLE  = "#A78BFA"
T1      = "#EDF2F7"
T2      = "#94A3B8"
T3      = "#556070"
BORDER  = "#252E42"


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
def card_head(parent, title: str, subtitle: str = "") -> tk.Frame:
    h = tk.Frame(parent, bg=parent["bg"])
    h.pack(fill="x", pady=(0, 10))
    tk.Label(h, text=title, bg=parent["bg"], fg=T1,
             font=("Arial", 11, "bold")).pack(side="left")
    if subtitle:
        tk.Label(h, text=subtitle, bg=parent["bg"], fg=T3,
                 font=("Arial", 8)).pack(side="left", padx=8)
    return h


def scrollable_frame(parent) -> tuple[tk.Canvas, tk.Frame]:
    """Área scrollable. Devuelve (canvas, inner_frame)."""
    canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
    sb     = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    inner  = tk.Frame(canvas, bg=BG)

    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    wid = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(wid, width=e.width))
    canvas.configure(yscrollcommand=sb.set)
    canvas.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    canvas.bind_all("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    return canvas, inner


def make_treeview(parent, columns: list[tuple], height: int = 15) -> ttk.Treeview:
    """
    Crea un Treeview oscuro con scrollbar.

    columns: lista de tuplas (id, cabecera, ancho) o (id, cabecera, ancho, anchor).
    anchor por defecto: "center" para columnas cortas, "w" para nombres.
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
        tree.column(col_id, width=width, minwidth=30, anchor=anchor)

    sb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
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