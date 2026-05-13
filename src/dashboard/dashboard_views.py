"""
dashboard_views.py  (v4)
========================
Cambios v4:
  · TODAS las queries usan 'fecha_acceso' (nombre real en la BD).
  · AccesosView: columnas reducidas en compact_mode para caber en 440 px.
    En compact se muestran: #, Nombre, Cuenta, Rol, Fecha, Hora.
    En normal se muestran todas (num, nombre, cuenta, grado, grupo, rol, fecha, hora).
  · StatsView: las dos cards superiores se apilan verticalmente en compact_mode
    (evita que cada card quede en < 220 px).
  · ResumenView: KPIs siguen en fila; el font de valor se redujo levemente
    en compact para caber bien en ~440 px.
  · ConfigView y PerfilView: sin cambios de lógica, respetan compact_mode.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import json, os, re

from dash_theme import (
    BG, SIDEBAR, CARD, CARD2, ACCENT, RED, AMBER, BLUE,
    T1, T2, T3, BORDER,
    card_head, scrollable_frame, make_treeview, iniciales,
)
from lang_dict import t

# ══════════════════════════════════════════════════════════════════════
#  CAPA DE DATOS UNIFICADA
# ══════════════════════════════════════════════════════════════════════
from data_source import (
    kpi_dentro      as _kpi_dentro,
    kpi_hoy         as _kpi_hoy,
    kpi_semana      as _kpi_semana,
    ultimos_accesos as _ultimos_accesos,
    accesos_todos   as _accesos_todos,
    top7            as _top7,
    stats_rol       as _stats_rol,
    stats_hora      as _stats_hora,
    perfil_stats    as _perfil_stats,
    fuente_activa,
    ultima_persona_acceso as _ultima_persona,
)

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "lab_config.json")


# ═══════════════════════════════════════════════════════════════
#  HORARIO (JSON)
# ═══════════════════════════════════════════════════════════════

def _load_horario() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"hora_inicio": "00:00", "hora_fin": "00:00"}


def _save_horario(inicio: str, fin: str):
    data = _load_horario()
    data.update({"hora_inicio": inicio, "hora_fin": fin})
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
#  BASE VIEW
# ═══════════════════════════════════════════════════════════════

class BaseView(tk.Frame):
    def __init__(self, parent, dashboard):
        super().__init__(parent, bg=BG)
        self.dash         = dashboard
        self.compact_mode = getattr(dashboard, "compact_mode", False)
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
#  RESUMEN VIEW
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


# ═══════════════════════════════════════════════════════════════
#  STATS VIEW
# ═══════════════════════════════════════════════════════════════

class StatsView(BaseView):
    def __init__(self, parent, dashboard):
        super().__init__(parent, dashboard)
        self._build()

    def _build(self):
        _, inner = scrollable_frame(self)
        pad = self._pad()
        cp  = self.compact_mode

        # ── En compact: las dos cards superiores van en columna ──
        # En normal:  van en fila (side-by-side)
        if cp:
            top_container = tk.Frame(inner, bg=BG)
            top_container.pack(fill="x", padx=pad, pady=(12, 0))

            self._c_top = tk.Frame(top_container, bg=CARD, padx=12, pady=10)
            self._c_top.pack(fill="x", pady=(0, 8))
            card_head(self._c_top, t("top_visitantes"), t("top_sub"), compact=True)
            self._top_inner = tk.Frame(self._c_top, bg=CARD)
            self._top_inner.pack(fill="both", expand=True)

            self._c_rol = tk.Frame(top_container, bg=CARD, padx=12, pady=10)
            self._c_rol.pack(fill="x")
            card_head(self._c_rol, t("accesos_por_rol"), t("dist_visitas"), compact=True)
            self._rol_inner = tk.Frame(self._c_rol, bg=CARD)
            self._rol_inner.pack(fill="both", expand=True)
        else:
            top_row = tk.Frame(inner, bg=BG)
            top_row.pack(fill="x", padx=pad, pady=(14, 0))

            self._c_top = tk.Frame(top_row, bg=CARD, padx=14, pady=12)
            self._c_top.pack(side="left", fill="both", expand=True, padx=(0, 8))
            card_head(self._c_top, t("top_visitantes"), t("top_sub"), compact=False)
            self._top_inner = tk.Frame(self._c_top, bg=CARD)
            self._top_inner.pack(fill="both", expand=True)

            self._c_rol = tk.Frame(top_row, bg=CARD, padx=14, pady=12)
            self._c_rol.pack(side="left", fill="both", expand=True)
            card_head(self._c_rol, t("accesos_por_rol"), t("dist_visitas"), compact=False)
            self._rol_inner = tk.Frame(self._c_rol, bg=CARD)
            self._rol_inner.pack(fill="both", expand=True)

        # Card: Por hora (siempre en ancho completo)
        self._c_hora = tk.Frame(inner, bg=CARD,
                                padx=12 if cp else 14,
                                pady=10 if cp else 12)
        self._c_hora.pack(fill="x", padx=pad, pady=14)
        card_head(self._c_hora, t("accesos_por_hora"), t("franja_horaria"), compact=cp)
        self._hora_inner = tk.Frame(self._c_hora, bg=CARD)
        self._hora_inner.pack(fill="both", expand=True)

    def refresh(self):
        self._render_top(_top7())
        self._render_rol(_stats_rol())
        self._render_horas(_stats_hora())

    def _render_top(self, rows):
        for w in self._top_inner.winfo_children():
            w.destroy()
        if not rows:
            tk.Label(self._top_inner, text=t("sin_datos"), bg=CARD, fg=T3,
                     font=("Arial", 9)).pack(pady=12)
            return
        medal = ["🥇", "🥈", "🥉"]
        cp = self.compact_mode
        for i, d in enumerate(rows):
            row = tk.Frame(self._top_inner, bg=CARD2,
                           padx=8 if cp else 10, pady=5 if cp else 7)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=medal[i] if i < 3 else f"  {i+1}.",
                     bg=CARD2, fg=T1,
                     font=("Arial", 9 if cp else 10)
                     ).pack(side="left", padx=(0, 6))
            info = tk.Frame(row, bg=CARD2)
            info.pack(side="left", fill="x", expand=True)
            tk.Label(info, text=d["nombre"], bg=CARD2, fg=T1,
                     font=("Arial", 8 if cp else 9, "bold"),
                     anchor="w").pack(anchor="w")
            tk.Label(info, text=d["sub"], bg=CARD2, fg=T3,
                     font=("Arial", 7 if cp else 8),
                     anchor="w").pack(anchor="w")
            tk.Label(row,
                     text=f"{d['visitas']} {t('col_visitas')}",
                     bg=CARD2, fg=ACCENT,
                     font=("Arial", 8 if cp else 9, "bold")
                     ).pack(side="right")

    def _render_rol(self, rows):
        for w in self._rol_inner.winfo_children():
            w.destroy()
        if not rows:
            tk.Label(self._rol_inner, text=t("sin_datos"), bg=CARD, fg=T3,
                     font=("Arial", 9)).pack(pady=12)
            return
        total  = sum(r[1] for r in rows) or 1
        colors = {"admin": ACCENT, "maestro": BLUE, "estudiante": AMBER}
        cp = self.compact_mode
        for rol, cnt in rows:
            pct = int(cnt / total * 100)
            row = tk.Frame(self._rol_inner, bg=CARD)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=rol.capitalize(), bg=CARD, fg=T2,
                     font=("Arial", 8 if cp else 9),
                     width=10 if cp else 12, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row, bg=CARD2, height=12 if cp else 14)
            bar_bg.pack(side="left", fill="x", expand=True, padx=6)
            bar_bg.pack_propagate(False)
            tk.Frame(bar_bg, bg=colors.get(rol, ACCENT),
                     width=max(4, pct * 2)).pack(side="left", fill="y")
            tk.Label(row, text=f"{cnt} ({pct}%)", bg=CARD, fg=T2,
                     font=("Arial", 8 if cp else 9)).pack(side="right")

    def _render_horas(self, rows):
        for w in self._hora_inner.winfo_children():
            w.destroy()
        if not rows:
            tk.Label(self._hora_inner, text=t("sin_datos_semana"),
                     bg=CARD, fg=T3, font=("Arial", 9)).pack(pady=12)
            return
        max_cnt = max(r[1] for r in rows) or 1
        bar_h   = 60 if self.compact_mode else 80
        bar_w   = 18 if self.compact_mode else 22
        canvas  = tk.Frame(self._hora_inner, bg=CARD)
        canvas.pack(fill="x")
        for hr, cnt in rows:
            pct   = max(4, int(cnt / max_cnt * bar_h))
            col_f = tk.Frame(canvas, bg=CARD, padx=2)
            col_f.pack(side="left", fill="y")
            tk.Frame(col_f, bg=CARD, height=bar_h - pct).pack(fill="x")
            tk.Frame(col_f, bg=ACCENT if cnt == max_cnt else BLUE,
                     height=pct, width=bar_w).pack()
            tk.Label(col_f, text=f"{hr:02d}", bg=CARD, fg=T3,
                     font=("Arial", 6 if self.compact_mode else 7)).pack()
            tk.Label(col_f, text=str(cnt), bg=CARD, fg=T2,
                     font=("Arial", 6 if self.compact_mode else 7)).pack()


# ═══════════════════════════════════════════════════════════════
#  PERFIL VIEW
# ═══════════════════════════════════════════════════════════════

class PerfilView(BaseView):
    def __init__(self, parent, dashboard):
        super().__init__(parent, dashboard)
        self._build()

    def _build(self):
        _, inner = scrollable_frame(self)
        pad = self._pad()
        cp  = self.compact_mode
        u   = self.dash.usuario
        ini = iniciales(u.get("nombre", ""), u.get("apellido_paterno", ""))

        c = tk.Frame(inner, bg=CARD,
                     padx=18 if cp else 24,
                     pady=14 if cp else 20)
        c.pack(fill="x", padx=pad, pady=14)

        av_font = ("Arial", 24 if cp else 32, "bold")
        tk.Label(c, text=ini, bg=ACCENT, fg=BG,
                 font=av_font, width=3, height=1).pack(pady=(0, 10))

        nombre_full = (
            f"{u.get('nombre', '')} "
            f"{u.get('apellido_paterno', '')} "
            f"{(u.get('apellido_materno') or '')}").strip()
        tk.Label(c, text=nombre_full, bg=CARD, fg=T1,
                 font=("Arial", 13 if cp else 16, "bold")).pack()
        rol_key = {"admin": "administrador", "maestro": "maestro"}.get(
            u["rol"], "alumno")
        tk.Label(c, text=t(rol_key), bg=CARD, fg=ACCENT,
                 font=("Arial", 9 if cp else 10)).pack(pady=(2, 12))

        grid = tk.Frame(c, bg=CARD)
        grid.pack()
        for lbl_key, val in [
            ("num_cuenta", u.get("numero_cuenta") or "—"),
            ("correo",     u.get("correo")        or "—"),
            ("col_rol",    u.get("rol", "").capitalize()),
        ]:
            row = tk.Frame(grid, bg=CARD)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"{t(lbl_key)}:", bg=CARD, fg=T3,
                     font=("Arial", 8 if cp else 9),
                     width=16 if cp else 18, anchor="e").pack(side="left")
            tk.Label(row, text=val, bg=CARD, fg=T1,
                     font=("Arial", 8 if cp else 9, "bold"),
                     anchor="w").pack(side="left", padx=8)

        c2 = tk.Frame(inner, bg=CARD,
                      padx=18 if cp else 24,
                      pady=12 if cp else 16)
        c2.pack(fill="x", padx=pad, pady=(0, 14))
        card_head(c2, t("actividad_mes"), "", compact=cp)
        self._stat_frame = tk.Frame(c2, bg=CARD)
        self._stat_frame.pack(fill="x")

    def refresh(self):
        for w in self._stat_frame.winfo_children():
            w.destroy()
        accesos, reg = _perfil_stats()
        cp = self.compact_mode
        for lbl_key, val in [("accesos_mes", accesos), ("usuarios_mes", reg)]:
            row = tk.Frame(self._stat_frame, bg=CARD2,
                           padx=10 if cp else 14, pady=8 if cp else 10)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=str(val), bg=CARD2, fg=ACCENT,
                     font=("Arial", 18 if cp else 22, "bold")
                     ).pack(side="left", padx=(0, 10))
            tk.Label(row, text=t(lbl_key), bg=CARD2, fg=T2,
                     font=("Arial", 8 if cp else 9)).pack(side="left")


# ═══════════════════════════════════════════════════════════════
#  CONFIG VIEW
# ═══════════════════════════════════════════════════════════════

class ConfigView(BaseView):
    def __init__(self, parent, dashboard):
        super().__init__(parent, dashboard)
        self._build()

    def _build(self):
        _, inner = scrollable_frame(self)
        pad = self._pad()
        cp  = self.compact_mode

        c = tk.Frame(inner, bg=CARD,
                     padx=14 if cp else 20,
                     pady=12 if cp else 16)
        c.pack(fill="x", padx=pad, pady=14)
        card_head(c, t("config_sistema"), t("prefs"), compact=cp)

        f = tk.Frame(c, bg=CARD)
        f.pack(fill="x")

        # Horario de acceso
        self._sep(f, "horario_acceso", "horario_desc")
        hr_row = tk.Frame(f, bg=CARD2, padx=10 if cp else 12, pady=8 if cp else 10)
        hr_row.pack(fill="x", pady=(4, 0))
        self._lbl_horario = tk.Label(
            hr_row, text=self._horario_text(), bg=CARD2, fg=T2,
            font=("Arial", 8 if cp else 9), anchor="w")
        self._lbl_horario.pack(side="left", fill="x", expand=True)
        tk.Button(hr_row, text=t("btn_config_horario"),
                  command=self._open_horario,
                  bg=ACCENT, fg=BG, relief="flat",
                  font=("Arial", 8 if cp else 9, "bold"),
                  padx=10, pady=4, cursor="hand2").pack(side="right")

        # Auto-actualización
        self._sep(f, "autoactualizacion", "autoactualizacion_desc")
        ar = tk.Frame(f, bg=CARD2, padx=10 if cp else 12, pady=6 if cp else 8)
        ar.pack(fill="x", pady=(4, 0))
        tk.Label(ar, text=t("activo_30s"), bg=CARD2, fg=ACCENT,
                 font=("Arial", 8 if cp else 9)).pack(anchor="w")

        # Info sesión
        self._sep(f, "info_sesion", "info_sesion_desc")
        ir = tk.Frame(f, bg=CARD2, padx=10 if cp else 12, pady=8 if cp else 10)
        ir.pack(fill="x", pady=(4, 0))
        u = self.dash.usuario
        for lbl_k, val in [
            ("col_nombre", f"{u.get('nombre', '')} {u.get('apellido_paterno', '')}"),
            ("num_cuenta", u.get("numero_cuenta") or "—"),
            ("correo",     u.get("correo") or "—"),
            ("col_rol",    u.get("rol", "").capitalize()),
        ]:
            tk.Label(ir, text=f"{t(lbl_k)}: {val}", bg=CARD2, fg=T2,
                     font=("Arial", 8 if cp else 9), anchor="w").pack(anchor="w")

    def _sep(self, parent, title_key, desc_key):
        cp = self.compact_mode
        tk.Label(parent, text=t(title_key), bg=CARD, fg=T1,
                 font=("Arial", 9 if cp else 10, "bold"),
                 anchor="w").pack(fill="x", pady=(14, 0))
        tk.Label(parent, text=t(desc_key), bg=CARD, fg=T3,
                 font=("Arial", 7 if cp else 8),
                 anchor="w").pack(fill="x")

    def _horario_text(self) -> str:
        cfg = _load_horario()
        ini = cfg.get("hora_inicio", "00:00")
        fin = cfg.get("hora_fin",    "00:00")
        fin_txt = t("sin_limite") if fin == "00:00" else fin
        return t("horario_actual", ini=ini, fin=fin_txt)

    def _open_horario(self):
        HorarioDialog(self.dash.root, on_save=lambda: self._lbl_horario.configure(
            text=self._horario_text()))

    def on_show(self):
        self._lbl_horario.configure(text=self._horario_text())


# ═══════════════════════════════════════════════════════════════
#  DIÁLOGO HORARIO
# ═══════════════════════════════════════════════════════════════

class HorarioDialog(tk.Toplevel):
    def __init__(self, parent, on_save=None):
        super().__init__(parent)
        self.on_save = on_save
        self.title(t("horario_titulo"))
        self.resizable(False, False)
        self.configure(bg=CARD)
        self.transient(parent)

        cfg      = _load_horario()
        self._ini = tk.StringVar(value=cfg.get("hora_inicio", "07:00"))
        self._fin = tk.StringVar(value=cfg.get("hora_fin",    "22:00"))
        self._err = tk.StringVar(value="")

        self._build()

        W_DLG, H_DLG = 380, 310
        px = parent.winfo_rootx() + parent.winfo_width()  // 2 - W_DLG // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - H_DLG // 2
        self.geometry(f"{W_DLG}x{H_DLG}+{px}+{py}")
        self.grab_set()
        self.lift()
        self.focus_force()

    def _build(self):
        pad = {"padx": 24, "pady": 6}
        tk.Label(self, text=t("horario_titulo"), bg=CARD, fg=T1,
                 font=("Arial", 13, "bold")).pack(pady=(18, 4))
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24)

        form = tk.Frame(self, bg=CARD)
        form.pack(fill="x", **pad)
        for lbl_key, var in [("hora_inicio", self._ini), ("hora_fin", self._fin)]:
            tk.Label(form, text=t(lbl_key), bg=CARD, fg=T2,
                     font=("Arial", 9), anchor="w").pack(fill="x", pady=(10, 0))
            tk.Entry(form, textvariable=var, bg=CARD2, fg=T1,
                     insertbackground=T1, relief="flat",
                     font=("Arial", 12), justify="center"
                     ).pack(fill="x", ipady=6)

        tk.Label(self, text="HH:MM  (ej: 07:00 — 21:00)",
                 bg=CARD, fg=T3, font=("Arial", 8)).pack(**pad)
        tk.Label(self, textvariable=self._err, bg=CARD, fg=RED,
                 font=("Arial", 8)).pack(padx=24)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24)
        brow = tk.Frame(self, bg=CARD)
        brow.pack(fill="x", padx=24, pady=(8, 18))
        tk.Button(brow, text=t("cancelar"), command=self.destroy,
                  bg=CARD2, fg=T2, relief="flat", font=("Arial", 9),
                  padx=12, pady=6, cursor="hand2").pack(side="left")
        tk.Button(brow, text=t("aplicar"), command=self._save,
                  bg=ACCENT, fg=BG, relief="flat", font=("Arial", 9, "bold"),
                  padx=16, pady=6, cursor="hand2").pack(side="right")

    def _save(self):
        ini = self._ini.get().strip()
        fin = self._fin.get().strip()
        if not re.fullmatch(r"\d{2}:\d{2}", ini) or \
           not re.fullmatch(r"\d{2}:\d{2}", fin):
            self._err.set(t("err_hora"))
            return
        _save_horario(ini, fin)
        messagebox.showinfo(t("exito"), t("horario_guardado"), parent=self)
        if self.on_save:
            self.on_save()
        self.destroy()