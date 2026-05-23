import os
import platform
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, Optional

from dash_theme import (
    aplicar_estilo_treeview, iniciales, rounded_card, modern_button,
)
from lang_dict import t, toggle_lang, fecha_local

# ── Paleta institucional UdeC ──────────────────────────────────
BG           = "#F5F0E8"   # beige institucional (fondo general)
SIDEBAR      = "#1B2A4A"   # azul navy (sidebar)
SIDEBAR_TEXT = "#FFFFFF"   # blanco para texto sobre navy
CARD         = "#EAE5D8"   # beige claro (cards/campos)
CARD2        = "#DDD8CB"   # beige medio (hover / filas alternas)
ACCENT       = "#006644"   # verde loro institucional UdeC
ACCENT2      = "#008855"   # verde loro medio
RED          = "#C1121F"   # rojo institucional
AMBER        = "#E07A00"   # naranja advertencia
BLUE         = "#1B2A4A"   # azul marino institucional UdeC
T1           = "#1A1A2E"   # texto oscuro casi negro
T2           = "#5C6170"   # gris medio (subtextos)
T3           = "#8A8FA0"   # gris claro (hints)
BORDER       = "#C8C2B2"   # borde beige-gris
BORDER_SB    = "#2E4068"   # borde dentro del sidebar navy
HDR_BG       = "#FFFFFF"   # blanco para el header (independiente del sidebar)


class Dashboard:

    def __init__(self, usuario: dict):
        self.usuario = usuario
        self.rol     = usuario["rol"]

        self.root = tk.Tk()
        self.root.title("LabControl")

        # ── Detección de Raspberry Pi ────────────────────────────
        self.es_rpi = self._detectar_rpi()
        print(f"[SCREEN] Raspberry Pi detectada: {self.es_rpi}")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        print(f"[SCREEN] Resolución: {screen_w}x{screen_h}")

        # ── Modo responsivo ──────────────────────────────────────
        if self.es_rpi or (screen_w <= 1024 and screen_h <= 600):
            # Raspberry Pi 7" — 1024×600 landscape
            self.compact_mode  = True
            self.rpi_mode      = True
            self.sidebar_width = 170
        elif screen_h > screen_w:
            # Portrait en otro dispositivo
            self.compact_mode  = True
            self.rpi_mode      = False
            self.sidebar_width = 160
        elif screen_w <= 1366:
            # Laptop / pantalla mediana
            self.compact_mode  = False
            self.rpi_mode      = False
            self.sidebar_width = 220
        else:
            # Desktop / pantalla grande
            self.compact_mode  = False
            self.rpi_mode      = False
            self.sidebar_width = 260

        # ── Tamaño y posición de ventana ────────────────────────
        if self.rpi_mode:
            # En RPi: pantalla completa exacta sin decoraciones de WM
            self.root.geometry("1024x600+0+0")
            self.root.overrideredirect(True)   # sin barra de título del SO
            self.root.resizable(False, False)
        else:
            self.root.minsize(800, 500)
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
        self.root.after(1_000, self._monitor_señal_refresh)
        self.root.after(600, lambda: self.navigate("inicio"))
        self.root.after(5_000, self._ciclo_refresh)

    @staticmethod
    def _detectar_rpi() -> bool:
        """Devuelve True si el programa corre sobre una Raspberry Pi."""
        # Método 1: archivo de modelo del device-tree (más fiable)
        for ruta in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
            try:
                with open(ruta, "r", errors="ignore") as f:
                    if "Raspberry Pi" in f.read():
                        return True
            except OSError:
                pass
        # Método 2: /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", "r") as f:
                contenido = f.read()
                if "Raspberry Pi" in contenido or "BCM" in contenido:
                    return True
        except OSError:
            pass
        # Método 3: arquitectura ARM en Linux
        if platform.system() == "Linux" and platform.machine().startswith(("arm", "aarch")):
            return True
        return False

        self.root.configure(bg=BG)

        aplicar_estilo_treeview()
        self._build_ui()
        self.root.after(1_000, self._monitor_señal_refresh)
        self.root.after(600, lambda: self.navigate("inicio"))
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

        # ── Logo institucional ────────────────────────────────────
        logo_px   = 12 if self.rpi_mode else (16 if self.compact_mode else 22)
        logo_pady = (8, 6) if self.rpi_mode else ((16, 10) if self.compact_mode else (26, 18))

        # Buscar carpeta img automáticamente y listar archivos para diagnóstico
        _here    = os.path.dirname(os.path.abspath(__file__))
        _img_dir = os.path.normpath(os.path.join(_here, "..", "img"))
        print(f"[LOGO] Carpeta img: {_img_dir}")
        print(f"[LOGO] Existe carpeta: {os.path.isdir(_img_dir)}")
        if os.path.isdir(_img_dir):
            print(f"[LOGO] Archivos: {os.listdir(_img_dir)}")

        # Intentar múltiples variantes del nombre (espacio, mayúsculas, etc.)
        LOGO_PATH = None
        for _n in ["UdeC_2LC_Blanco.png", "UdeC_2LC _Blanco.png",
                   "UdeC_2LC_blanco.png", "UdeC_2LC _blanco.png"]:
            _c = os.path.join(_img_dir, _n)
            if os.path.isfile(_c):
                LOGO_PATH = _c
                print(f"[LOGO] Encontrado: {_c}")
                break

        if not LOGO_PATH:
            print("[LOGO] ⚠️  Archivo no encontrado en img/")

        logo_frame = tk.Frame(sb, bg=SIDEBAR)
        logo_frame.pack(fill="x", padx=logo_px, pady=logo_pady)

        max_w = self.sidebar_width - logo_px * 2
        max_h = 80 if self.rpi_mode else (130 if self.compact_mode else 170)

        if LOGO_PATH:
            try:
                from PIL import Image, ImageTk
                img = Image.open(LOGO_PATH).convert("RGBA")
                img.thumbnail((max_w, max_h), Image.LANCZOS)
                self._logo_img = ImageTk.PhotoImage(img)
                tk.Label(logo_frame, image=self._logo_img,
                         bg=SIDEBAR, borderwidth=0).pack()
                print("[LOGO] Cargado con Pillow ✓")
            except Exception as e_pil:
                print(f"[LOGO Pillow] {e_pil}")
                try:
                    self._logo_img = tk.PhotoImage(file=LOGO_PATH)
                    iw, ih = self._logo_img.width(), self._logo_img.height()
                    factor = max(1, max(iw // max(max_w, 1), ih // max(max_h, 1)))
                    if factor > 1:
                        self._logo_img = self._logo_img.subsample(factor, factor)
                    tk.Label(logo_frame, image=self._logo_img,
                             bg=SIDEBAR, borderwidth=0).pack()
                    print("[LOGO] Cargado con tk.PhotoImage ✓")
                except Exception as e_tk:
                    print(f"[LOGO tkinter] {e_tk}")

        tk.Frame(sb, bg=BORDER_SB, height=1).pack(fill="x", padx=logo_px, pady=(12, 0))

        self._build_profile(sb)
        tk.Frame(sb, bg=BORDER_SB, height=1).pack(fill="x", padx=logo_px, pady=(8, 0))

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
            sz   =  9 if self.compact_mode else 10
            pady = (5, 1) if self.compact_mode else (12, 3)
            padx = 10 if self.compact_mode else 14
            tk.Label(nav, text=t(key).upper(), bg=SIDEBAR, fg=SIDEBAR_TEXT,
                     font=("Segoe UI", sz, "bold"),
                     anchor="w").pack(fill="x", padx=padx, pady=pady)

        grp("nav_panel")
        self._nav("inicio",   "🏠 Inicio",        nav)

        grp("nav_gestion")
        self._nav("alumnos",  t("nav_alumnos"),  nav)
        if self.rol == "admin":
            self._nav("maestros", t("nav_maestros"), nav)
            self._nav("admins",   t("nav_admins"),   nav)

        # Espaciador al final para que el scroll funcione cuando hay poco contenido
        tk.Frame(nav, bg=SIDEBAR, height=30).pack(fill="x")

        # ── Footer fijo (reloj + logout) ─────────────────────────
        tk.Frame(sb, bg=BORDER_SB, height=1).pack(fill="x", side="bottom")

        footer = tk.Frame(sb, bg=SIDEBAR)
        footer.pack(side="bottom", fill="x",
                    padx=12 if self.compact_mode else 16,
                    pady=(8, 14))

        logout_btn = modern_button(
            footer, text="⬅  Cerrar sesión", command=self.logout,
            bg_color="#C1121F", fg_color=SIDEBAR_TEXT,
            font_size=9 if self.compact_mode else 11,
            padding=(10, 7) if self.compact_mode else (14, 9))
        logout_btn.pack(fill="x")

    def _build_profile(self, sb):
        padx = 12 if self.compact_mode else 18
        pady = 8  if self.compact_mode else 14

        f = tk.Frame(sb, bg=SIDEBAR)
        f.pack(fill="x", padx=padx, pady=pady)

        ini = iniciales(
            self.usuario.get("nombre", ""),
            self.usuario.get("apellido_paterno", ""))
        av_sz  = 11 if self.compact_mode else 14
        av_pad =  5 if self.compact_mode else  8

        # Avatar: fondo blanco semitransparente simulado con ACCENT2
        avatar_frame = tk.Frame(f, bg=SIDEBAR_TEXT)
        avatar_frame.pack(side="left", padx=(0, 8 if self.compact_mode else 11))
        tk.Label(avatar_frame, text=ini, bg=SIDEBAR_TEXT, fg=BLUE,
                 font=("Segoe UI", av_sz, "bold"), width=3, height=1
                 ).pack(padx=av_pad, pady=av_pad)

        info = tk.Frame(f, bg=SIDEBAR)
        info.pack(side="left", fill="x", expand=True)

        max_len = 18 if self.compact_mode else 26
        nombre_full = (
            f"{self.usuario.get('nombre', '')} "
            f"{self.usuario.get('apellido_paterno', '')}").strip()[:max_len]

        nf = (("Segoe UI", 10, "bold") if self.compact_mode
              else ("Segoe UI", 11, "bold"))
        rf = (("Segoe UI",  8) if self.compact_mode
              else ("Segoe UI", 10))

        tk.Label(info, text=nombre_full, bg=SIDEBAR, fg=SIDEBAR_TEXT,
                 font=nf, anchor="w").pack(anchor="w")
        rol_key = {"admin": "administrador", "maestro": "maestro"}.get(
            self.rol, "alumno")
        tk.Label(info, text=t(rol_key), bg=SIDEBAR, fg=ACCENT2,
                 font=rf, anchor="w").pack(anchor="w")

        # Todo el bloque de perfil abre el modal al hacer clic
        for w in f.winfo_children() + [f]:
            w.bind("<Button-1>", lambda e: self._show_perfil_popup())
            w.configure(cursor="hand2")

    def _nav(self, view_id: str, label: str, parent):
        btn_frame = tk.Frame(parent, bg=SIDEBAR)
        btn_frame.pack(fill="x", pady=1)

        font_sz = 11 if self.compact_mode else 12
        padx    = 10 if self.compact_mode else 14
        pady    =  4 if self.compact_mode else  8

        # Hover suave: azul navy un tono más claro
        NAV_HOVER = "#2E4068"

        btn = tk.Button(
            btn_frame, text=f" {label}", anchor="w",
            bg=SIDEBAR, fg=SIDEBAR_TEXT,
            activebackground=NAV_HOVER, activeforeground=SIDEBAR_TEXT,
            font=("Segoe UI", font_sz), relief="flat", cursor="hand2",
            padx=padx, pady=pady,
            command=lambda v=view_id: self.navigate(v))
        btn.pack(fill="x")
        self._nav_btns[view_id] = btn

    def _build_header(self, parent):
        hdr_h = 50 if self.rpi_mode else (62 if self.compact_mode else 72)
        hdr   = tk.Frame(parent, bg=HDR_BG, height=hdr_h)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Frame(hdr, bg=BORDER, height=1).pack(side="bottom", fill="x")

        left = tk.Frame(hdr, bg=HDR_BG)
        left.pack(side="left",
                  padx=12 if self.compact_mode else 22,
                  pady= 8 if self.compact_mode else 14)

        tf = ("Segoe UI", 15, "bold") if self.compact_mode else ("Segoe UI", 19, "bold")

        self.hdr_title = tk.Label(left, text=t("title_resumen"),
                                  bg=HDR_BG, fg=BLUE, font=tf)
        self.hdr_title.pack(anchor="w")

        right = tk.Frame(hdr, bg=HDR_BG)
        right.pack(side="right",
                   padx=12 if self.compact_mode else 18,
                   pady= 6 if self.compact_mode else 10)

        # ── Botón de traductor ────────────────────────────────────
        self._btn_lang = modern_button(
            right, text=t("btn_traductor"),
            command=self._toggle_language,
            bg_color=BLUE, fg_color=HDR_BG,
            font_size=9 if self.compact_mode else 10,
            padding=(10, 5) if self.compact_mode else (14, 7))
        self._btn_lang.pack(side="right")

    # ── Vistas ───────────────────────────────────────────────────
    def _build_views(self, parent):
        from dashboard_views   import InicioView, AccesosView
        from dashboard_gestion import GestionView

        self.content = tk.Frame(parent, bg=BG)
        self.content.pack(fill="both", expand=True)

        self.views: Dict = {
            "inicio":  InicioView(self.content,  self),
            "accesos": AccesosView(self.content, self),
            "alumnos": GestionView(self.content, self, "estudiante"),
        }
        if self.rol == "admin":
            self.views["maestros"] = GestionView(self.content, self, "maestro")
            self.views["admins"]   = GestionView(self.content, self, "admin")

        self._current: str = ""

    # ─────────────────────────────────────────────────────────────
    # NAVEGACIÓN
    # ─────────────────────────────────────────────────────────────
    _META_KEYS = {
        "inicio":   ("title_inicio",   "sub_inicio"),
        "accesos":  ("title_accesos",  "sub_accesos"),
        "alumnos":  ("title_alumnos",  "sub_alumnos"),
        "maestros": ("title_maestros", "sub_maestros"),
        "admins":   ("title_admins",   "sub_admins"),
    }

    _HDR_LABELS = {
        "inicio":   ("Dashboard", ""),
        "accesos":  None,
        "alumnos":  None,
        "maestros": None,
        "admins":   None,
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
        NAV_HOVER = "#2E4068"
        nav_sz = 11 if self.compact_mode else 12
        for vid, btn in self._nav_btns.items():
            if vid == view_id:
                btn.configure(bg=NAV_HOVER, fg=SIDEBAR_TEXT,
                              font=("Segoe UI", nav_sz, "bold"))
            else:
                btn.configure(bg=SIDEBAR, fg=SIDEBAR_TEXT,
                              font=("Segoe UI", nav_sz, "normal"))

        keys = self._META_KEYS.get(view_id, ("title_resumen", "sub_resumen"))
        hardcoded = self._HDR_LABELS.get(view_id)
        if hardcoded:
            self.hdr_title.configure(text=hardcoded[0])
        else:
            self.hdr_title.configure(text=t(keys[0]))

    # ─────────────────────────────────────────────────────────────
    # TRADUCTOR
    # ─────────────────────────────────────────────────────────────
    def _toggle_language(self):
        toggle_lang()
        prev = self._current
        self._build_ui()
        self.navigate(prev if prev else "inicio")

    # ─────────────────────────────────────────────────────────────
    # AUTO-REFRESH
    # ─────────────────────────────────────────────────────────────
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
    # POPUP DE PERFIL
    # ─────────────────────────────────────────────────────────────
    def _show_perfil_popup(self):
        u   = self.usuario
        ini = iniciales(u.get("nombre", ""), u.get("apellido_paterno", ""))
        nombre_full = (
            f"{u.get('nombre', '')} "
            f"{u.get('apellido_paterno', '')} "
            f"{(u.get('apellido_materno') or '')}").strip()

        pop = tk.Toplevel(self.root)
        pop.title("Perfil de usuario")
        pop.resizable(False, False)
        pop.configure(bg=HDR_BG)
        pop.transient(self.root)

        W, H = 380, 360
        px = self.root.winfo_rootx() + self.root.winfo_width()  // 2 - W // 2
        py = self.root.winfo_rooty() + self.root.winfo_height() // 2 - H // 2
        pop.geometry(f"{W}x{H}+{px}+{py}")
        pop.grab_set()
        pop.lift()
        pop.focus_force()

        # Avatar
        tk.Label(pop, text=ini, bg=ACCENT, fg=HDR_BG,
                 font=("Segoe UI", 28, "bold"),
                 width=3, height=1).pack(pady=(28, 10),
                                         ipadx=12, ipady=12)

        # Nombre y rol
        tk.Label(pop, text=nombre_full, bg=HDR_BG, fg=BLUE,
                 font=("Segoe UI", 14, "bold")).pack()
        rol_display = {"admin": "Administrador",
                       "maestro": "Maestro"}.get(u.get("rol", ""), "Alumno")
        tk.Label(pop, text=rol_display, bg=HDR_BG, fg=ACCENT,
                 font=("Segoe UI", 11)).pack(pady=(3, 14))

        tk.Frame(pop, bg=BORDER, height=1).pack(fill="x", padx=28)

        # Datos
        grid = tk.Frame(pop, bg=HDR_BG)
        grid.pack(fill="x", padx=28, pady=14)
        for lbl, val in [
            ("No. Cuenta", u.get("numero_cuenta") or "—"),
            ("Correo",     u.get("correo")        or "—"),
            ("Rol",        rol_display),
        ]:
            row = tk.Frame(grid, bg=HDR_BG)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=f"{lbl}:", bg=HDR_BG, fg=T3,
                     font=("Segoe UI", 10), width=12, anchor="e").pack(side="left")
            tk.Label(row, text=val, bg=HDR_BG, fg=BLUE,
                     font=("Segoe UI", 10, "bold"), anchor="w"
                     ).pack(side="left", padx=10)

        tk.Frame(pop, bg=BORDER, height=1).pack(fill="x", padx=28)

        # Cerrar haciendo clic fuera del modal
        pop.bind("<FocusOut>", lambda e: pop.destroy() if not str(e.widget).startswith(str(pop)) else None)
        pop.bind("<Escape>", lambda e: pop.destroy())

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