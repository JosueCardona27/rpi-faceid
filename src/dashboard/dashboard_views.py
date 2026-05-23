import tkinter as tk
from tkinter import ttk
from datetime import datetime
import os

from dash_theme import (
    card_head, scrollable_frame, make_treeview, iniciales,
)
from lang_dict import t, toggle_lang, fecha_local

# ── Paleta institucional UdeC ──────────────────────────────────
BG      = "#F5F0E8"
SIDEBAR = "#FFFFFF"
CARD    = "#EAE5D8"
CARD2   = "#DDD8CB"
ACCENT  = "#006644"
ACCENT2 = "#008855"
RED     = "#C1121F"
AMBER   = "#E07A00"
BLUE    = "#1B2A4A"
T1      = "#1A1A2E"
T2      = "#5C6170"
T3      = "#8A8FA0"
BORDER  = "#C8C2B2"

# ══════════════════════════════════════════════════════════════════════
#  CAPA DE DATOS UNIFICADA
# ══════════════════════════════════════════════════════════════════════
from data_source import (
    kpi_dentro      as _kpi_dentro,
    kpi_hoy         as _kpi_hoy,
    kpi_semana      as _kpi_semana,
    ultimos_accesos as _ultimos_accesos,
    accesos_todos   as _accesos_todos,
    fuente_activa,
    ultima_persona_acceso as _ultima_persona,
)


# ═══════════════════════════════════════════════════════════════
#  BASE VIEW
# ═══════════════════════════════════════════════════════════════

class BaseView(tk.Frame):
    def __init__(self, parent, dashboard):
        super().__init__(parent, bg=BG)
        self.dash         = dashboard
        self.compact_mode = getattr(dashboard, "compact_mode", False)
        self.rpi_mode     = getattr(dashboard, "rpi_mode",     False)
        self.sidebar_width = getattr(dashboard, "sidebar_width", 250)

    def on_show(self):
        self.refresh()

    def refresh(self):
        pass

    def _pad(self, normal=18, compact=12):
        """Padding lateral según modo."""
        return compact if self.compact_mode else normal

    def _fsz(self, normal=10, compact=8):
        """Tamaño de fuente según modo."""
        return compact if self.compact_mode else normal


# ═══════════════════════════════════════════════════════════════
#  INICIO VIEW  (reemplaza Resumen + Registro en un solo apartado)
# ═══════════════════════════════════════════════════════════════

class InicioView(BaseView):
    """
    Vista principal del Dashboard.
    Muestra:
      · Tarjeta de aforo: personas dentro vs límite de 40.
      · Tabla completa de registros de entrada con roles coloreados.
    """
    LIMITE = 40

    def __init__(self, parent, dashboard):
        super().__init__(parent, dashboard)
        self._build()

    # ── Construcción ────────────────────────────────────────
    def _build(self):
        _, inner = scrollable_frame(self)
        pad = self._pad()
        cp  = self.compact_mode
        rpi = self.rpi_mode

        # ── Tarjeta de aforo ─────────────────────────────────
        aforo_card = tk.Frame(inner, bg=CARD,
                              padx=10 if rpi else (14 if cp else 20),
                              pady= 6 if rpi else (12 if cp else 16))
        aforo_card.pack(fill="x", padx=pad, pady=(6 if rpi else (12 if cp else 16), 6))

        head_af = tk.Frame(aforo_card, bg=CARD)
        head_af.pack(fill="x", pady=(0, 6))
        card_head(head_af, "Aforo actual",
                  f"Límite permitido: {self.LIMITE} alumnos", compact=True)

        body_af = tk.Frame(aforo_card, bg=CARD2,
                           padx=10 if rpi else (14 if cp else 20),
                           pady= 6 if rpi else (10 if cp else 14))
        body_af.pack(fill="x")

        left_af = tk.Frame(body_af, bg=CARD2)
        left_af.pack(side="left")

        num_sz = 22 if rpi else (34 if cp else 46)
        sep_sz = 14 if rpi else (20 if cp else 26)

        self._lbl_dentro = tk.Label(
            left_af, text="—", bg=CARD2, fg=ACCENT,
            font=("Segoe UI", num_sz, "bold"))
        self._lbl_dentro.pack(side="left")

        tk.Label(left_af, text=f" / {self.LIMITE}",
                 bg=CARD2, fg=T3,
                 font=("Segoe UI", sep_sz)).pack(side="left", padx=(0, 10))

        right_af = tk.Frame(body_af, bg=CARD2)
        right_af.pack(side="left", fill="y", padx=(6, 0))
        tk.Label(right_af, text="personas", bg=CARD2, fg=BLUE,
                 font=("Segoe UI", 9 if rpi else (10 if cp else 11), "bold")).pack(anchor="w")
        tk.Label(right_af, text="en el laboratorio", bg=CARD2, fg=T2,
                 font=("Segoe UI", 8 if rpi else (9 if cp else 10))).pack(anchor="w")

        bar_outer = tk.Frame(aforo_card, bg=CARD)
        bar_outer.pack(fill="x", pady=(5, 0))
        self._bar_bg = tk.Frame(bar_outer, bg=CARD2, height=5)
        self._bar_bg.pack(fill="x")
        self._bar_bg.pack_propagate(False)
        self._bar_fill = tk.Frame(self._bar_bg, bg=ACCENT, height=5)
        self._bar_fill.pack(side="left", fill="y")

        # ── Tabla de registros ───────────────────────────────
        c_table = tk.Frame(inner, bg=CARD,
                           padx=8 if rpi else (12 if cp else 14),
                           pady=6 if rpi else (10 if cp else 12))
        c_table.pack(fill="both", expand=True, padx=pad, pady=(0, 8))

        hdr = tk.Frame(c_table, bg=CARD)
        hdr.pack(fill="x", pady=(0, 4))
        card_head(hdr, "Registro de entradas",
                  "Historial de accesos al laboratorio", compact=True)

        self._lbl_count = tk.Label(c_table, text="", bg=CARD, fg=BLUE,
                                    font=("Arial", 9 if rpi else (10 if cp else 11), "bold"))
        self._lbl_count.pack(anchor="w", pady=(0, 3))

        cols = [
            ("num",    "#",                          42, "center"),
            ("nombre", "Nombre completo", 160 if rpi else (180 if cp else 230), "w"),
            ("cuenta", "No. Cuenta",     110 if rpi else (130 if cp else 155), "center"),
            ("grado",  "Grado",           60 if rpi else ( 75 if cp else 100), "center"),
            ("grupo",  "Grupo",           55 if rpi else ( 70 if cp else  95), "center"),
            ("rol",    "Rol",             85 if rpi else (100 if cp else 130), "center"),
            ("fecha",  "Fecha",          100 if rpi else (115 if cp else 145), "center"),
            ("hora",   "Hora",            65 if rpi else ( 80 if cp else 100), "center"),
        ]
        tree_h = 9 if rpi else (18 if cp else 22)

        wrap = tk.Frame(c_table, bg=CARD)
        wrap.pack(fill="both", expand=True)
        self.tree = make_treeview(wrap, cols, height=tree_h, xscroll=True)

        # Colores por rol (tags en ttk.Treeview)
        self.tree.tag_configure("estudiante", foreground=BLUE)
        self.tree.tag_configure("maestro",    foreground=AMBER)
        self.tree.tag_configure("admin",      foreground=ACCENT)

    # ── Refresh ─────────────────────────────────────────────
    def refresh(self):
        dentro = _kpi_dentro()
        pct    = min(dentro / self.LIMITE, 1.0)

        # Color dinámico según nivel de ocupación
        if pct >= 1.0:
            color = RED
        elif pct >= 0.8:
            color = AMBER
        else:
            color = ACCENT

        self._lbl_dentro.configure(text=str(dentro), fg=color)
        self._bar_fill.configure(bg=color)

        # Actualizar barra de progreso
        def _actualizar_barra():
            bw = self._bar_bg.winfo_width()
            if bw > 1:
                fw = max(4, int(pct * bw))
                self._bar_fill.configure(width=fw)

        self._bar_bg.after(50, _actualizar_barra)

        # Tabla de registros
        rows = _accesos_todos("")
        self._lbl_count.configure(
            text=f"{len(rows)} registro(s) encontrado(s)")

        self.tree.delete(*self.tree.get_children())
        if not rows:
            self.tree.insert("", "end",
                             values=("—", "Sin registros",
                                     "", "", "", "", "", ""))
            return

        for r in rows:
            rol_raw = (r.get("rol") or "").lower()
            tag = rol_raw if rol_raw in ("estudiante", "maestro", "admin") else ""
            self.tree.insert(
                "", "end",
                values=(r["num"], r["nombre"], r["cuenta"],
                        r["grado"], r["grupo"], r["rol"],
                        r["fecha"], r["hora"]),
                tags=(tag,) if tag else ())


# ═══════════════════════════════════════════════════════════════
#  RESUMEN VIEW  (mantenida internamente; ya no aparece en nav)
# ═══════════════════════════════════════════════════════════════

class ResumenView(BaseView):
    def __init__(self, parent, dashboard):
        super().__init__(parent, dashboard)
        self._build()

    def _build(self):
        _, inner = scrollable_frame(self)
        pad = self._pad()
        cp  = self.compact_mode

        # ── KPIs ──────────────────────────────────────────────
        kpi_row = tk.Frame(inner, bg=BG)
        kpi_row.pack(fill="x", padx=pad, pady=(8 if cp else 12, 14))
        self._kv: dict[str, tk.StringVar] = {}

        for key, lbl_key, color in [
            ("dentro", "kpi_inside", ACCENT),
            ("hoy",    "kpi_hoy",    BLUE),
            ("semana", "kpi_semana", AMBER),
        ]:
            self._kv[key] = tk.StringVar(value="—")
            self._kpi_card(kpi_row, t(lbl_key), key, color)

        # ── Última persona que entró ───────────────────────────
        cp_pad = 10 if cp else 14
        c_ult = tk.Frame(inner, bg=CARD, padx=cp_pad, pady=cp_pad)
        c_ult.pack(fill="x", padx=pad, pady=(0, 10))

        hult = tk.Frame(c_ult, bg=CARD)
        hult.pack(fill="x", pady=(0, 6))
        card_head(hult, "Última entrada", "Persona más reciente", compact=cp)

        self._ult_frame = tk.Frame(c_ult, bg=CARD)
        self._ult_frame.pack(fill="x", pady=(0, 6))

        # ── Últimos 4 accesos ──────────────────────────────────
        c4 = tk.Frame(inner, bg=CARD, padx=cp_pad, pady=cp_pad)
        c4.pack(fill="x", padx=pad, pady=(0, 10))
        h4 = tk.Frame(c4, bg=CARD)
        h4.pack(fill="x", pady=(0, 6))
        card_head(h4, t("ultimos_accesos"), t("accesos_rt"), compact=cp)

        fnt_lbl = self._fsz(8, 7)
        tk.Label(h4, text=t("en_vivo"), bg=CARD, fg=ACCENT,
                 font=("Segoe UI", fnt_lbl, "bold")).pack(side="right")
        self._fuente_lbl = tk.Label(h4, text="", bg=CARD,
                                     font=("Segoe UI", max(fnt_lbl - 1, 6)))
        self._fuente_lbl.pack(side="right", padx=4)

        self._frame4 = tk.Frame(c4, bg=CARD)
        self._frame4.pack(fill="x", pady=(0, 6))

        # ── Tabla de accesos recientes ─────────────────────────
        c_rec = tk.Frame(inner, bg=CARD, padx=cp_pad, pady=cp_pad)
        c_rec.pack(fill="x", padx=pad, pady=(0, 20))
        hrec = tk.Frame(c_rec, bg=CARD)
        hrec.pack(fill="x", pady=(0, 6))
        card_head(hrec, t("accesos_recientes"), t("movimientos_hoy"), compact=cp)

        tk.Button(hrec, text=t("ver_todo"),
                  command=lambda: self.dash.navigate("accesos"),
                  bg=CARD2, fg=T2, relief="flat",
                  font=("Segoe UI", self._fsz(8, 7)),
                  padx=6, pady=2, cursor="hand2").pack(side="right")

        if cp:
            cols = [
                ("nombre", t("col_persona"), 160, "w"),
                ("tipo",   t("col_tipo"),     65, "center"),
                ("hora",   t("col_hora"),     55, "center"),
                ("rol",    t("col_rol"),      60, "center"),
            ]
            tree_h = 5
        else:
            cols = [
                ("nombre", t("col_persona"), 260, "w"),
                ("tipo",   t("col_tipo"),    100, "center"),
                ("hora",   t("col_hora"),     80, "center"),
                ("rol",    t("col_rol"),     100, "center"),
            ]
            tree_h = 8

        wrap = tk.Frame(c_rec, bg=CARD)
        wrap.pack(fill="x", pady=(0, 6))
        self.tree_rec = make_treeview(wrap, cols, height=tree_h)

    def _kpi_card(self, parent, label, key, color):
        cp = self.compact_mode
        f  = tk.Frame(parent, bg=CARD,
                      padx=8 if cp else 16,
                      pady=8 if cp else 14)
        f.pack(side="left", fill="both", expand=True, padx=(0, 6 if cp else 10))

        val_font = ("Segoe UI", 22, "bold") if cp else ("Segoe UI", 28, "bold")
        lbl_font = ("Segoe UI",  8)         if cp else ("Segoe UI",  9)

        tk.Label(f, textvariable=self._kv[key], bg=CARD, fg=color,
                 font=val_font).pack(anchor="w")
        tk.Label(f, text=label, bg=CARD, fg=T2,
                 font=lbl_font).pack(anchor="w")

    def refresh(self):
        self._kv["dentro"].set(str(_kpi_dentro()))
        self._kv["hoy"].set(str(_kpi_hoy()))
        self._kv["semana"].set(str(_kpi_semana()))

        # Indicador de fuente
        fuente = fuente_activa()
        if fuente == "api":
            self._fuente_lbl.configure(text="🟢 Raspberry Pi", fg="#22C55E")
        else:
            self._fuente_lbl.configure(text="🟡 BD local",     fg="#F59E0B")

        # Cards de últimos 4 accesos
        for w in self._frame4.winfo_children():
            w.destroy()
        ultimos = _ultimos_accesos(4)
        cp = self.compact_mode
        if not ultimos:
            tk.Label(self._frame4, text=t("nadie_lab"),
                     bg=CARD, fg=T3,
                     font=("Segoe UI", self._fsz(9, 8))).pack(pady=10)
        else:
            for p in ultimos:
                row = tk.Frame(self._frame4, bg=CARD2,
                               padx=6 if cp else 8, pady=5 if cp else 7)
                row.pack(fill="x", pady=1)
                partes = p["nombre"].split(" ", 1)
                ini    = iniciales(partes[0], partes[1] if len(partes) > 1 else "")

                tk.Label(row, text=ini, bg=ACCENT, fg=BG,
                         font=("Segoe UI", 8 if cp else 9, "bold"),
                         width=2 if cp else 3
                         ).pack(side="left", padx=(0, 6))
                info = tk.Frame(row, bg=CARD2)
                info.pack(side="left", fill="x", expand=True)
                tk.Label(info, text=p["nombre"], bg=CARD2, fg=T1,
                         font=("Segoe UI", 8 if cp else 9),
                         anchor="w").pack(anchor="w")
                tk.Label(info, text=p["rol"], bg=CARD2, fg=T3,
                         font=("Segoe UI", 7 if cp else 8)).pack(side="right")
                tipo_txt = t("entrada") if p["tipo"] == "entrada" else t("salida")
                tk.Label(row, text=f"{tipo_txt}  {p['hora']}",
                         bg=CARD2, fg=T2,
                         font=("Arial", 8)).pack(side="right")

        # Tabla de accesos recientes
        self.tree_rec.delete(*self.tree_rec.get_children())
        for r in _ultimos_accesos(8):
            tipo_txt = t("entrada") if r["tipo"] == "entrada" else t("salida")
            self.tree_rec.insert("", "end",
                                  values=(r["nombre"], tipo_txt,
                                          r["hora"], r["rol"]))

        # Tarjeta de última entrada
        for w in self._ult_frame.winfo_children():
            w.destroy()
        ultima = _ultima_persona()
        if not ultima:
            tk.Label(self._ult_frame,
                     text="Aún no hay registros de entrada.",
                     bg=CARD, fg=T3, font=("Arial", 9)).pack(pady=8)
            return

        cp = self.compact_mode
        rp = 8 if cp else 12
        row_u = tk.Frame(self._ult_frame, bg=CARD2, padx=rp, pady=rp)
        row_u.pack(fill="x")

        partes = ultima["nombre"].split(" ", 1)
        ini_u  = iniciales(partes[0], partes[1] if len(partes) > 1 else "")

        tk.Label(row_u, text=ini_u, bg=ACCENT, fg=BG,
                 font=("Segoe UI", 10 if cp else 12, "bold"),
                 width=3 if cp else 4
                 ).pack(side="left", padx=(0, 7))

        info_u = tk.Frame(row_u, bg=CARD2)
        info_u.pack(side="left", fill="x", expand=True)
        tk.Label(info_u, text=ultima["nombre"], bg=CARD2, fg=T1,
                 font=("Segoe UI", 9 if cp else 10, "bold"),
                 anchor="w").pack(anchor="w")
        tk.Label(info_u,
                 text=f"{ultima['rol'].capitalize()}  ·  {ultima['cuenta']}",
                 bg=CARD2, fg=T3,
                 font=("Segoe UI", 7 if cp else 8),
                 anchor="w").pack(anchor="w")

        cnt_f = tk.Frame(row_u, bg=CARD2)
        cnt_f.pack(side="right", padx=(6, 0))
        tk.Label(cnt_f, text=str(ultima["visitas_total"]),
                 bg=CARD2, fg=ACCENT,
                 font=("Arial", 20 if cp else 22, "bold")).pack()
        tk.Label(cnt_f, text="visitas", bg=CARD2, fg=T3,
                 font=("Arial", 7 if cp else 8)).pack()

        tk.Label(row_u,
                 text=f"{ultima['fecha']}  {ultima['hora']}",
                 bg=CARD2, fg=T2, font=("Arial", 7 if cp else 8)
                 ).pack(side="right", padx=(0, 10))


# ═══════════════════════════════════════════════════════════════
#  ACCESOS VIEW
# ═══════════════════════════════════════════════════════════════

class AccesosView(BaseView):
    def __init__(self, parent, dashboard):
        super().__init__(parent, dashboard)
        self._rol_opts = [
            ("",           t("filtro_todos")),
            ("estudiante", t("filtro_estudiante")),
            ("maestro",    t("filtro_maestro")),
            ("admin",      t("filtro_admin")),
        ]
        self._display_to_rol = {v: k for k, v in self._rol_opts}
        self._cb_var = tk.StringVar(value=t("filtro_todos"))
        self._build()

    def _build(self):
        pad = self._pad()
        cp  = self.compact_mode

        c = tk.Frame(self, bg=CARD, padx=12 if cp else 14, pady=10 if cp else 12)
        c.pack(fill="both", expand=True, padx=pad, pady=pad)

        hdr = tk.Frame(c, bg=CARD)
        hdr.pack(fill="x", pady=(0, 8))
        card_head(hdr, t("registro_completo"), t("hist_entradas"), compact=cp)

        # Filtro por rol
        frow = tk.Frame(hdr, bg=CARD)
        frow.pack(side="right")
        tk.Label(frow, text=t("filtro_rol"), bg=CARD, fg=T2,
                 font=("Arial", 8 if cp else 9)).pack(side="left")
        self._rol_cb = ttk.Combobox(
            frow, textvariable=self._cb_var,
            values=[v for _, v in self._rol_opts],
            state="readonly", width=11 if cp else 13)
        self._rol_cb.pack(side="left", padx=(4, 0))
        self._rol_cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh())

        self._lbl_count = tk.Label(c, text="", bg=CARD, fg=T2,
                                    font=("Arial", 8 if cp else 9))
        self._lbl_count.pack(anchor="w", pady=(0, 5))

        # ── Columnas ──────────────────────────────────────────────────────
        # En compact_mode se activa scroll horizontal para ver todas las columnas
        # sin recortar datos. Las 8 columnas completas siguen disponibles.
        if cp:
            cols = [
                ("num",    t("col_num"),    40,  "center"),
                ("nombre", t("col_nombre"), 160, "w"),
                ("cuenta", t("col_cuenta"), 100, "center"),
                ("grado",  t("col_grado"),   55, "center"),
                ("grupo",  t("col_grupo"),   50, "center"),
                ("rol",    t("col_rol"),     75, "center"),
                ("fecha",  t("col_fecha"),   85, "center"),
                ("hora",   t("col_hora"),    55, "center"),
            ]
            tree_h = 20
        else:
            cols = [
                ("num",    t("col_num"),    45,  "center"),
                ("nombre", t("col_nombre"), 200, "w"),
                ("cuenta", t("col_cuenta"), 110, "center"),
                ("grado",  t("col_grado"),   65, "center"),
                ("grupo",  t("col_grupo"),   55, "center"),
                ("rol",    t("col_rol"),     90, "center"),
                ("fecha",  t("col_fecha"),   95, "center"),
                ("hora",   t("col_hora"),    65, "center"),
            ]
            tree_h = 24

        self._compact_cols = cp  # recordar para refresh()

        wrap = tk.Frame(c, bg=CARD)
        wrap.pack(fill="both", expand=True)
        # xscroll=True en compact: el usuario desliza horizontalmente para ver todo
        self.tree = make_treeview(wrap, cols, height=tree_h, xscroll=cp)

    def refresh(self):
        texto_sel  = self._cb_var.get()
        rol_filtro = self._display_to_rol.get(texto_sel, "")
        rows       = _accesos_todos(rol_filtro)

        self._lbl_count.configure(
            text=t("registros_encontrados", n=len(rows)))

        self.tree.delete(*self.tree.get_children())
        if not rows:
            # 8 columnas siempre (compact ahora muestra todas con xscroll)
            self.tree.insert("", "end",
                             values=("—", t("sin_registros"), "", "", "", "", "", ""))
            return

        for r in rows:
            # Ambos modos insertan las 8 columnas completas
            self.tree.insert("", "end", values=(
                r["num"], r["nombre"], r["cuenta"],
                r["grado"], r["grupo"], r["rol"],
                r["fecha"], r["hora"],
            ))