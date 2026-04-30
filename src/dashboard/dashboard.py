"""
dashboard.py  (v5)
==================
Ventana principal del dashboard LabControl.

Cambios v5:
  - Eliminados abrir_registro_facial_persona() y abrir_registro():
    la captura biométrica ahora la gestiona CapturaBiometricaDialog
    directamente desde dashboard_gestion.py (sin subprocess).
"""

import tkinter as tk
from tkinter import messagebox
from datetime import datetime

from dash_theme import (
    BG, SIDEBAR, CARD, CARD2, ACCENT, RED, T1, T2, T3, BORDER,
    aplicar_estilo_treeview, iniciales,
)
from lang_dict import t, toggle_lang, fecha_local

class Dashboard:

    def __init__(self, usuario: dict):
        self.usuario = usuario
        self.rol     = usuario["rol"]

        self.root = tk.Tk()
        self.root.title("LabControl")
        self.root.geometry("1280x780")
        self.root.minsize(980, 640)
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
        sb = tk.Frame(parent, bg=SIDEBAR, width=230)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        tk.Label(sb, text="LabControl", bg=SIDEBAR, fg=ACCENT,
                 font=("Arial", 17, "bold")).pack(
                     anchor="w", padx=18, pady=(20, 0))
        tk.Label(sb, text=t("sistema_gestion"), bg=SIDEBAR, fg=T3,
                 font=("Arial", 8)).pack(
                     anchor="w", padx=18, pady=(0, 12))
        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x")

        self._build_profile(sb)
        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x")

        self._nav_btns: dict[str, tk.Button] = {}
        nav = tk.Frame(sb, bg=SIDEBAR)
        nav.pack(fill="x", pady=6)

        def grp(key):
            tk.Label(nav, text=t(key).upper(), bg=SIDEBAR, fg=T3,
                     font=("Arial", 7, "bold"),
                     anchor="w").pack(fill="x", padx=18, pady=(12, 2))

        grp("nav_panel")
        self._nav("resumen", t("nav_resumen"), nav)
        self._nav("accesos", t("nav_accesos"), nav)
        self._nav("stats",   t("nav_stats"),   nav)

        grp("nav_gestion")
        self._nav("alumnos", t("nav_alumnos"), nav)
        if self.rol == "admin":
            self._nav("maestros", t("nav_maestros"), nav)
            self._nav("admins",   t("nav_admins"),   nav)

        grp("nav_cuenta")
        self._nav("perfil", t("nav_perfil"), nav)
        self._nav("config", t("nav_config"), nav)

        footer = tk.Frame(sb, bg=SIDEBAR)
        footer.pack(side="bottom", fill="x", padx=14, pady=14)
        self.clock_lbl = tk.Label(
            footer, text="--:--:--", bg=SIDEBAR, fg=T2,
            font=("Courier", 12, "bold"))
        self.clock_lbl.pack(anchor="w", pady=(0, 10))
        tk.Button(footer, text=t("cerrar_sesion"), command=self.logout,
                  bg=RED, fg="white", relief="flat",
                  font=("Arial", 9),
                  cursor="hand2", padx=10, pady=8).pack(fill="x")

    def _build_profile(self, sb):
        f = tk.Frame(sb, bg=SIDEBAR, pady=12)
        f.pack(fill="x", padx=14)
        ini = iniciales(
            self.usuario.get("nombre", ""),
            self.usuario.get("apellido_paterno", ""))
        tk.Label(f, text=ini, bg=ACCENT, fg=BG,
                 font=("Arial", 12, "bold"), width=3,
                 height=1).pack(side="left", padx=(0, 10))
        info = tk.Frame(f, bg=SIDEBAR)
        info.pack(side="left")
        nombre_full = (
            f"{self.usuario.get('nombre', '')} "
            f"{self.usuario.get('apellido_paterno', '')}").strip()[:24]
        tk.Label(info, text=nombre_full, bg=SIDEBAR, fg=T1,
                 font=("Arial", 9, "bold"),
                 anchor="w").pack(anchor="w")
        rol_key = {"admin": "administrador",
                   "maestro": "maestro"}.get(self.rol, "alumno")
        tk.Label(info, text=t(rol_key), bg=SIDEBAR, fg=ACCENT,
                 font=("Arial", 8), anchor="w").pack(anchor="w")

    def _nav(self, view_id: str, label: str, parent):
        btn = tk.Button(
            parent, text=label, anchor="w",
            bg=SIDEBAR, fg=T2,
            activebackground=CARD2, activeforeground=T1,
            font=("Arial", 9), relief="flat", cursor="hand2",
            padx=18, pady=9,
            command=lambda v=view_id: self.navigate(v))
        btn.pack(fill="x")
        self._nav_btns[view_id] = btn

    # ── Header ───────────────────────────────────────────────────
    def _build_header(self, parent):
        hdr = tk.Frame(parent, bg=SIDEBAR, height=60)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Frame(hdr, bg=BORDER, height=1).pack(side="bottom", fill="x")

        left = tk.Frame(hdr, bg=SIDEBAR)
        left.pack(side="left", padx=20, pady=6)
        self.hdr_title = tk.Label(
            left, text=t("title_resumen"),
            bg=SIDEBAR, fg=T1, font=("Arial", 13, "bold"))
        self.hdr_title.pack(anchor="w")
        self.hdr_sub = tk.Label(
            left, text=t("sub_resumen"),
            bg=SIDEBAR, fg=T3, font=("Arial", 8))
        self.hdr_sub.pack(anchor="w")

        self._btn_lang = tk.Button(
            hdr, text=t("btn_traductor"),
            command=self._toggle_language,
            bg=CARD2, fg=ACCENT, relief="flat",
            font=("Arial", 9, "bold"), cursor="hand2",
            padx=10, pady=6)
        self._btn_lang.pack(side="right", padx=12)

        self._hdr_fecha = tk.Label(
            hdr, text=fecha_local(),
            bg=SIDEBAR, fg=T2, font=("Arial", 9))
        self._hdr_fecha.pack(side="right", padx=8)

    # ── Vistas ───────────────────────────────────────────────────
    def _build_views(self, parent):
        from dashboard_views   import (ResumenView, AccesosView, StatsView, PerfilView, ConfigView)
        from dashboard_gestion import GestionView
    
        self.content = tk.Frame(parent, bg=BG)
        self.content.pack(fill="both", expand=True)

        self.views: dict = {
            "resumen": ResumenView(self.content, self),
            "accesos": AccesosView(self.content, self),
            "stats":   StatsView(self.content,   self),
            "alumnos": GestionView(self.content, self, "estudiante"),
            "perfil":  PerfilView(self.content,  self),
            "config":  ConfigView(self.content,  self),
        }
        if self.rol == "admin":
            self.views["maestros"] = GestionView(
                self.content, self, "maestro")
            self.views["admins"]   = GestionView(
                self.content, self, "admin")

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

        for vid, btn in self._nav_btns.items():
            btn.configure(
                bg=CARD   if vid == view_id else SIDEBAR,
                fg=ACCENT if vid == view_id else T2)

        keys = self._META_KEYS.get(
            view_id, ("title_resumen", "sub_resumen"))
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
        self.root.after(5_000, self._ciclo_refresh)  # ← era 30_000

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