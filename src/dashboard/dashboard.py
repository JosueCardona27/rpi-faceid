"""
dashboard.py  (v6)
==================
Ventana principal del dashboard LabControl.

Cambios v6:
  - Detección correcta de modo portrait (pantalla rotada 90°).
  - Maximizar ventana compatible con Windows Y Linux/Raspberry Pi.
  - minsize reducido a 500×600 para permitir 600px de ancho en portrait.
  - Eliminado doble-emoji en botones de navegación (las claves de lang_dict
    ya incluyen el emoji; el dict de icons extra fue eliminado).
  - Eliminado el pack duplicado del nav_scrollbar (causaba error en tkinter).
  - Spacer de relleno movido al FINAL de la lista de navegación, no al inicio.
  - font_size en navigate() ahora respeta compact_mode.
  - Scroll de la nav-canvas usa Enter/Leave (alineado con dash_theme).
"""

import platform
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime
from typing import Dict, Optional

from dash_theme import (
    BG, SIDEBAR, CARD, CARD2, ACCENT, RED, T1, T2, T3, BORDER,
    aplicar_estilo_treeview, iniciales, rounded_card, modern_button,
)
from lang_dict import t, toggle_lang, fecha_local


class Dashboard:

    def __init__(self, usuario: dict):
        self.usuario = usuario
        self.rol     = usuario["rol"]

        self.root = tk.Tk()
        self.root.title("LabControl")

        # ── Detección de pantalla ────────────────────────────────
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        print(f"[DEBUG] Pantalla: {screen_w}x{screen_h}")

        # Portrait = pantalla más alta que ancha (ej. 1024×600 rotada → 600×1024)
        portrait = screen_h > screen_w

        if portrait:
            # Pantalla vertical (Raspberry Pi con display rotado)
            self.compact_mode  = True
            self.sidebar_width = 160
        elif screen_w <= 1366:
            # Laptop / pantalla mediana
            self.compact_mode  = False
            self.sidebar_width = 220
        else:
            # Desktop / pantalla grande
            self.compact_mode  = False
            self.sidebar_width = 260

        # Mínimo reducido para soportar portrait de 600 px de ancho
        self.root.minsize(500, 600)

        # Maximizar: compatible con Windows y Linux/Raspberry Pi
        try:
            if platform.system() == "Windows":
                self.root.state("zoomed")
            else:
                self.root.attributes("-zoomed", True)
        except Exception:
            self.root.geometry(f"{screen_w}x{screen_h}")

        self.root.configure(bg=BG)

        aplicar_estilo_treeview()
        self._build_ui()
        self._start_clock()
        self.root.after(1_000, self._monitor_señal_refresh)
        self.root.after(600, lambda: self.navigate("resumen"))
        self.root.after(5_000, self._ciclo_refresh)

    # ─────────────────────────────────────────────────────────────
    # UI PRINCIPAL
    # ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        for w in self.root.winfo_children():
            w.destroy()

        root_frame = tk.Frame(self.root, bg=BG)
        root_frame.pack(fill="both", expand=True)

        self._build_sidebar(root_frame)
        self.main = tk.Frame(root_frame, bg=BG)
        self.main.pack(side="left", fill="both", expand=True)
        self._build_header(self.main)
        self._build_views(self.main)

    # ── Sidebar ──────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=SIDEBAR, width=self.sidebar_width)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        # Logo
        logo_px   = 16 if self.compact_mode else 22
        logo_pady = (16, 10) if self.compact_mode else (26, 18)

        logo_frame = tk.Frame(sb, bg=SIDEBAR)
        logo_frame.pack(fill="x", padx=logo_px, pady=logo_pady)

        icon_sz  = 20 if self.compact_mode else 24
        title_sz = 15 if self.compact_mode else 18
        sub_sz   =  8 if self.compact_mode else  9

        tk.Label(logo_frame, text="🧪", bg=SIDEBAR, fg=ACCENT,
                 font=("Segoe UI", icon_sz)).pack()
        tk.Label(logo_frame, text="LabControl", bg=SIDEBAR, fg=ACCENT,
                 font=("Segoe UI", title_sz, "bold")).pack(pady=(4, 2))
        tk.Label(logo_frame, text=t("sistema_gestion"), bg=SIDEBAR, fg=T3,
                 font=("Segoe UI", sub_sz)).pack()

        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", padx=logo_px, pady=(12, 0))

        self._build_profile(sb)
        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", padx=logo_px, pady=(8, 0))

        self._nav_btns: Dict[str, tk.Button] = {}

        # ── Canvas scrollable para navegación ───────────────────
        nav_container = tk.Frame(sb, bg=SIDEBAR)
        nav_container.pack(fill="both", expand=True,
                           padx=8 if self.compact_mode else 12, pady=2)

        nav_canvas    = tk.Canvas(nav_container, bg=SIDEBAR, highlightthickness=0)
        nav_scrollbar = ttk.Scrollbar(nav_container, orient="vertical",
                                       command=nav_canvas.yview)
        nav           = tk.Frame(nav_canvas, bg=SIDEBAR)

        def _cfg_nav_scroll(event=None):
            nav_canvas.configure(scrollregion=nav_canvas.bbox("all"))

        nav.bind("<Configure>", _cfg_nav_scroll)
        nav_win = nav_canvas.create_window((0, 0), window=nav, anchor="nw")

        def _cfg_nav_width(event=None):
            nav_canvas.itemconfig(nav_win, width=event.width)

        nav_canvas.bind("<Configure>", _cfg_nav_width)
        nav_canvas.configure(yscrollcommand=nav_scrollbar.set)

        nav_scrollbar.pack(side="right", fill="y")
        nav_canvas.pack(side="left", fill="both", expand=True)

        # Scroll de la barra nav sólo cuando el ratón está dentro
        def _nav_mw(event):
            nav_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _nav_lnx(event):
            nav_canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

        def _nav_enter(e=None):
            nav_canvas.bind_all("<MouseWheel>", _nav_mw)
            nav_canvas.bind_all("<Button-4>",   _nav_lnx)
            nav_canvas.bind_all("<Button-5>",   _nav_lnx)

        def _nav_leave(e=None):
            nav_canvas.unbind_all("<MouseWheel>")
            nav_canvas.unbind_all("<Button-4>")
            nav_canvas.unbind_all("<Button-5>")

        nav_canvas.bind("<Enter>", _nav_enter)
        nav_canvas.bind("<Leave>", _nav_leave)
        nav.bind("<Enter>", _nav_enter)
        nav.bind("<Leave>", _nav_leave)

        # ── Grupos de navegación ─────────────────────────────────
        def grp(key):
            sz   =  7 if self.compact_mode else  8
            pady = (5, 1) if self.compact_mode else (12, 3)
            padx = 10 if self.compact_mode else 14
            tk.Label(nav, text=t(key).upper(), bg=SIDEBAR, fg=T3,
                     font=("Segoe UI", sz, "bold"),
                     anchor="w").pack(fill="x", padx=padx, pady=pady)

        grp("nav_panel")
        self._nav("resumen",  t("nav_resumen"),  nav)
        self._nav("accesos",  t("nav_accesos"),  nav)
        self._nav("stats",    t("nav_stats"),     nav)

        grp("nav_gestion")
        self._nav("alumnos",  t("nav_alumnos"),  nav)
        if self.rol == "admin":
            self._nav("maestros", t("nav_maestros"), nav)
            self._nav("admins",   t("nav_admins"),   nav)

        grp("nav_cuenta")
        self._nav("perfil", t("nav_perfil"), nav)
        self._nav("config", t("nav_config"), nav)

        # Espaciador al final para que el scroll funcione cuando hay poco contenido
        tk.Frame(nav, bg=SIDEBAR, height=30).pack(fill="x")

        # ── Footer fijo (reloj + logout) ─────────────────────────
        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", side="bottom")

        footer = tk.Frame(sb, bg=SIDEBAR)
        footer.pack(side="bottom", fill="x",
                    padx=12 if self.compact_mode else 16,
                    pady=(8, 12))

        logout_btn = modern_button(
            footer, text=t("cerrar_sesion"), command=self.logout,
            bg_color=RED, fg_color="white",
            font_size=8 if self.compact_mode else 10,
            padding=(10, 6) if self.compact_mode else (14, 8))
        logout_btn.pack(fill="x", pady=(0, 8))

        clock_frame = tk.Frame(footer, bg=CARD)
        clock_frame.pack(fill="x")
        self.clock_lbl = tk.Label(
            clock_frame, text="--:--:--", bg=CARD, fg=ACCENT,
            font=("Courier", 12 if self.compact_mode else 14, "bold"))
        self.clock_lbl.pack(pady=6)

    def _build_profile(self, sb):
        padx = 12 if self.compact_mode else 18
        pady = 8  if self.compact_mode else 14

        f = tk.Frame(sb, bg=SIDEBAR)
        f.pack(fill="x", padx=padx, pady=pady)

        ini = iniciales(
            self.usuario.get("nombre", ""),
            self.usuario.get("apellido_paterno", ""))
        av_sz  =  9 if self.compact_mode else 13
        av_pad =  5 if self.compact_mode else  7

        avatar_frame = tk.Frame(f, bg=ACCENT)
        avatar_frame.pack(side="left", padx=(0, 8 if self.compact_mode else 11))
        tk.Label(avatar_frame, text=ini, bg=ACCENT, fg=BG,
                 font=("Segoe UI", av_sz, "bold"), width=3, height=1
                 ).pack(padx=av_pad, pady=av_pad)

        info = tk.Frame(f, bg=SIDEBAR)
        info.pack(side="left", fill="x", expand=True)

        max_len = 18 if self.compact_mode else 26
        nombre_full = (
            f"{self.usuario.get('nombre', '')} "
            f"{self.usuario.get('apellido_paterno', '')}").strip()[:max_len]

        nf = (("Segoe UI",  9, "bold") if self.compact_mode
              else ("Segoe UI", 10, "bold"))
        rf = (("Segoe UI",  7) if self.compact_mode
              else ("Segoe UI",  9))

        tk.Label(info, text=nombre_full, bg=SIDEBAR, fg=T1,
                 font=nf, anchor="w").pack(anchor="w")
        rol_key = {"admin": "administrador", "maestro": "maestro"}.get(
            self.rol, "alumno")
        tk.Label(info, text=t(rol_key), bg=SIDEBAR, fg=ACCENT,
                 font=rf, anchor="w").pack(anchor="w")

    def _nav(self, view_id: str, label: str, parent):
        """
        Crea un botón de navegación.
        Las etiquetas ya contienen emoji (vienen de lang_dict), no se duplican.
        """
        btn_frame = tk.Frame(parent, bg=SIDEBAR)
        btn_frame.pack(fill="x", pady=1)

        font_sz = 9 if self.compact_mode else 10
        padx    = 10 if self.compact_mode else 14
        pady    =  3 if self.compact_mode else  7

        btn = tk.Button(
            btn_frame, text=f" {label}", anchor="w",
            bg=SIDEBAR, fg=T2,
            activebackground=CARD2, activeforeground=T1,
            font=("Segoe UI", font_sz), relief="flat", cursor="hand2",
            padx=padx, pady=pady,
            command=lambda v=view_id: self.navigate(v))
        btn.pack(fill="x")
        self._nav_btns[view_id] = btn

    # ── Header ───────────────────────────────────────────────────
    def _build_header(self, parent):
        hdr_h = 56 if self.compact_mode else 68
        hdr   = tk.Frame(parent, bg=SIDEBAR, height=hdr_h)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Frame(hdr, bg=BORDER, height=1).pack(side="bottom", fill="x")

        left = tk.Frame(hdr, bg=SIDEBAR)
        left.pack(side="left",
                  padx=12 if self.compact_mode else 22,
                  pady= 8 if self.compact_mode else 14)

        tf = ("Segoe UI", 13, "bold") if self.compact_mode else ("Segoe UI", 16, "bold")
        sf = ("Segoe UI",  8)         if self.compact_mode else ("Segoe UI",  9)

        self.hdr_title = tk.Label(left, text=t("title_resumen"),
                                  bg=SIDEBAR, fg=T1, font=tf)
        self.hdr_title.pack(anchor="w")
        self.hdr_sub = tk.Label(left, text=t("sub_resumen"),
                                bg=SIDEBAR, fg=T3, font=sf)
        self.hdr_sub.pack(anchor="w", pady=(1, 0))

        right = tk.Frame(hdr, bg=SIDEBAR)
        right.pack(side="right",
                   padx=12 if self.compact_mode else 18,
                   pady= 6 if self.compact_mode else 10)

        self._btn_lang = modern_button(
            right, text=t("btn_traductor"),
            command=self._toggle_language,
            bg_color=CARD2, fg_color=ACCENT,
            font_size=8 if self.compact_mode else 9,
            padding=(8, 4) if self.compact_mode else (12, 6))
        self._btn_lang.pack(side="right", padx=(6, 0))

        # En compact la fecha no se muestra para no robar espacio al título
        if not self.compact_mode:
            self._hdr_fecha = tk.Label(
                right, text=fecha_local(),
                bg=SIDEBAR, fg=T2, font=("Segoe UI", 9))
            self._hdr_fecha.pack(side="right", padx=6)

    # ── Vistas ───────────────────────────────────────────────────
    def _build_views(self, parent):
        from dashboard_views   import ResumenView, AccesosView, StatsView, PerfilView, ConfigView
        from dashboard_gestion import GestionView

        self.content = tk.Frame(parent, bg=BG)
        self.content.pack(fill="both", expand=True)

        self.views: Dict = {
            "resumen": ResumenView(self.content, self),
            "accesos": AccesosView(self.content, self),
            "stats":   StatsView(self.content,   self),
            "alumnos": GestionView(self.content, self, "estudiante"),
            "perfil":  PerfilView(self.content,  self),
            "config":  ConfigView(self.content,  self),
        }
        if self.rol == "admin":
            self.views["maestros"] = GestionView(self.content, self, "maestro")
            self.views["admins"]   = GestionView(self.content, self, "admin")

        self._current: str = ""

    # ─────────────────────────────────────────────────────────────
    # NAVEGACIÓN
    # ─────────────────────────────────────────────────────────────
    _META_KEYS = {
        "resumen":  ("title_resumen",  "sub_resumen"),
        "accesos":  ("title_accesos",  "sub_accesos"),
        "stats":    ("title_stats",    "sub_stats"),
        "alumnos":  ("title_alumnos",  "sub_alumnos"),
        "maestros": ("title_maestros", "sub_maestros"),
        "admins":   ("title_admins",   "sub_admins"),
        "perfil":   ("title_perfil",   "sub_perfil"),
        "config":   ("title_config",   "sub_config"),
    }

    def navigate(self, view_id: str):
        if view_id not in self.views:
            return
        if view_id == self._current:
            self.views[view_id].refresh()
            return

        if self._current and self._current in self.views:
            self.views[self._current].pack_forget()

        v = self.views[view_id]
        v.pack(fill="both", expand=True)
        v.on_show()
        self._current = view_id

        # Resaltar botón activo (respeta tamaño de fuente del modo)
        nav_sz = 9 if self.compact_mode else 10
        for vid, btn in self._nav_btns.items():
            if vid == view_id:
                btn.configure(bg=CARD, fg=ACCENT,
                              font=("Segoe UI", nav_sz, "bold"))
            else:
                btn.configure(bg=SIDEBAR, fg=T2,
                              font=("Segoe UI", nav_sz, "normal"))

        keys = self._META_KEYS.get(view_id, ("title_resumen", "sub_resumen"))
        self.hdr_title.configure(text=t(keys[0]))
        self.hdr_sub.configure(text=t(keys[1]))

    # ─────────────────────────────────────────────────────────────
    # TRADUCTOR
    # ─────────────────────────────────────────────────────────────
    def _toggle_language(self):
        toggle_lang()
        prev = self._current
        self._build_ui()
        self.navigate(prev if prev else "resumen")

    # ─────────────────────────────────────────────────────────────
    # RELOJ Y AUTO-REFRESH
    # ─────────────────────────────────────────────────────────────
    def _start_clock(self):
        def tick():
            if self.root.winfo_exists():
                self.clock_lbl.configure(
                    text=datetime.now().strftime("%H:%M:%S"))
                self.root.after(1_000, tick)
        tick()

    def _ciclo_refresh(self):
        if not self.root.winfo_exists():
            return
        try:
            if self._current in self.views:
                self.views[self._current].refresh()
        except Exception as e:
            print(f"[REFRESH] {e}")
        self.root.after(5_000, self._ciclo_refresh)

    def _monitor_señal_refresh(self):
        """Detecta cambios en el archivo de señal y refresca al instante."""
        import os
        _ruta = os.path.join(os.path.dirname(__file__),
                             '..', 'database', '.refresh_signal')
        self._ultima_señal = 0.0

        def _check():
            if not self.root.winfo_exists():
                return
            try:
                mtime = os.path.getmtime(_ruta)
                if mtime != self._ultima_señal:
                    self._ultima_señal = mtime
                    if self._current in self.views:
                        self.views[self._current].refresh()
            except Exception:
                pass
            self.root.after(1_000, _check)

        _check()

    # ─────────────────────────────────────────────────────────────
    # LOGOUT
    # ─────────────────────────────────────────────────────────────
    def logout(self):
        if messagebox.askyesno(t("logout_titulo"),
                               t("confirmar_salida"),
                               parent=self.root):
            self.root.destroy()
            import login
            login.LoginWindow().run()

    def run(self):
        self.root.mainloop()