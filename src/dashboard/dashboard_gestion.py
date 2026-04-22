"""
dashboard_gestion.py  (v5)
==========================
Gestión de usuarios (alumnos, maestros, admins).

Cambios v5:
  - Captura biométrica integrada mediante CapturaBiometricaDialog
    (ya no se lanza interfaz.py como subprocess).
  - EstudianteFormDialog y PersonFormDialog abren el diálogo de captura
    facial dentro del propio dashboard al completar el formulario.
  - _lf() y _campo() son métodos de instancia (ya no colisionan).
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

from dash_theme import (
    BG, CARD, CARD2, ACCENT, RED, BLUE, T1, T2, T3, BORDER,
    card_head, make_treeview, iniciales,
)
from lang_dict import t


# ═══════════════════════════════════════════════════════════════════
#  CAPA DE DATOS
# ═══════════════════════════════════════════════════════════════════

def _db():
    from database import conectar
    return conectar()


def _listar(rol: str) -> list[dict]:
    try:
        conn = _db()
        c    = conn.cursor()
        try:
            c.execute("""
                SELECT u.id, u.nombre, u.apellido_paterno, u.apellido_materno,
                       u.numero_cuenta, u.correo, u.rol, u.fecha_registro,
                       ed.grado, ed.grupo,
                       ua.nombre || ' ' || ua.apellido_paterno AS act_por,
                       u.fecha_actualizacion
                FROM usuarios u
                LEFT JOIN estudiantes_detalle ed ON ed.usuario_id = u.id
                LEFT JOIN usuarios ua ON ua.id = u.actualizado_por
                WHERE u.rol = ?
                ORDER BY u.apellido_paterno, u.nombre
            """, (rol,))
        except Exception:
            # Fallback sin fecha_actualizacion / actualizado_por
            c.execute("""
                SELECT u.id, u.nombre, u.apellido_paterno, u.apellido_materno,
                       u.numero_cuenta, u.correo, u.rol, u.fecha_registro,
                       ed.grado, ed.grupo,
                       NULL AS act_por, NULL AS fecha_actualizacion
                FROM usuarios u
                LEFT JOIN estudiantes_detalle ed ON ed.usuario_id = u.id
                WHERE u.rol = ?
                ORDER BY u.apellido_paterno, u.nombre
            """, (rol,))

        rows = c.fetchall()
        conn.close()
        result = []
        for row in rows:
            uid, nom, ap, am, cuenta, correo, rol_u, freg, grado, grupo, act_por, fact = row
            result.append({
                "id":        uid,
                "nombre":    nom or "",
                "ap":        ap or "",
                "am":        "" if (not am or am == ".") else am,
                "cuenta":    cuenta or "",
                "correo":    correo or "",
                "rol":       rol_u or "",
                "fecha_reg": (freg or "")[:10],
                "grado":     grado,
                "grupo":     grupo or "",
                "act_por":   act_por or "—",
                "fecha_act": (fact or "")[:10] or "—",
            })
        return result
    except Exception as e:
        print(f"[LISTAR] {e}")
        return []


def _crear(datos: dict, reg_por: int) -> tuple[bool, str]:
    try:
        from database import registrar_usuario
        uid = registrar_usuario(
            nombre           = datos["nombre"],
            apellido_paterno = datos["ap"],
            apellido_materno = datos.get("am", ""),
            rol              = datos["rol"],
            numero_cuenta    = datos.get("cuenta"),
            correo           = datos.get("correo"),
            contrasena       = datos.get("contrasena"),
            grado            = datos.get("grado"),
            grupo            = datos.get("grupo"),
            registrado_por   = reg_por,
        )
        return (True, t("exito")) if uid and uid > 0 \
               else (False, "No se pudo registrar. Verifica los datos.")
    except Exception as e:
        return False, str(e)


def _editar(datos: dict, editor_id: int) -> tuple[bool, str]:
    try:
        import hashlib
        conn = _db()
        c    = conn.cursor()
        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sets   = ["nombre=?", "apellido_paterno=?", "apellido_materno=?",
                  "numero_cuenta=?", "fecha_actualizacion=?",
                  "actualizado_por=?"]
        params = [datos["nombre"], datos["ap"],
                  datos.get("am", "") or ".",
                  datos.get("cuenta"), now, editor_id]

        if datos.get("correo"):
            sets.append("correo=?")
            params.append(datos["correo"])
        if datos.get("contrasena"):
            h = hashlib.sha256(datos["contrasena"].encode()).hexdigest()
            sets.append("contrasena=?")
            params.append(h)
        params.append(datos["id"])

        try:
            c.execute(
                f"UPDATE usuarios SET {', '.join(sets)} WHERE id=?", params)
        except Exception:
            # Fallback sin fecha_actualizacion / actualizado_por
            sets2   = ["nombre=?", "apellido_paterno=?",
                       "apellido_materno=?", "numero_cuenta=?"]
            params2 = [datos["nombre"], datos["ap"],
                       datos.get("am", "") or ".", datos.get("cuenta")]
            if datos.get("correo"):
                sets2.append("correo=?")
                params2.append(datos["correo"])
            if datos.get("contrasena"):
                h = hashlib.sha256(datos["contrasena"].encode()).hexdigest()
                sets2.append("contrasena=?")
                params2.append(h)
            params2.append(datos["id"])
            c.execute(
                f"UPDATE usuarios SET {', '.join(sets2)} WHERE id=?", params2)

        if datos["rol"] == "estudiante" and datos.get("grado"):
            c.execute(
                "SELECT id FROM estudiantes_detalle WHERE usuario_id=?",
                (datos["id"],))
            if c.fetchone():
                c.execute(
                    "UPDATE estudiantes_detalle SET grado=?, grupo=? "
                    "WHERE usuario_id=?",
                    (int(datos["grado"]), datos.get("grupo", ""), datos["id"]))
            else:
                c.execute(
                    "INSERT INTO estudiantes_detalle (usuario_id, grado, grupo) "
                    "VALUES (?, ?, ?)",
                    (datos["id"], int(datos["grado"]), datos.get("grupo", "")))

        conn.commit()
        conn.close()
        return True, t("exito")
    except Exception as e:
        return False, str(e)


def _eliminar(uid: int) -> tuple[bool, str]:
    try:
        conn = _db()
        c    = conn.cursor()
        c.execute("DELETE FROM estudiantes_detalle WHERE usuario_id=?", (uid,))
        c.execute("DELETE FROM usuarios WHERE id=?", (uid,))
        conn.commit()
        conn.close()
        return True, t("exito")
    except Exception as e:
        return False, str(e)


# ═══════════════════════════════════════════════════════════════════
#  GESTION VIEW
# ═══════════════════════════════════════════════════════════════════

_META = {
    "estudiante": ("title_alumnos",  "sub_alumnos"),
    "maestro":    ("title_maestros", "sub_maestros"),
    "admin":      ("title_admins",   "sub_admins"),
}


class GestionView(tk.Frame):
    def __init__(self, parent, dashboard, rol_tipo: str):
        super().__init__(parent, bg=BG)
        self.dash     = dashboard
        self.rol_tipo = rol_tipo
        self._datos:  list[dict] = []
        self._build()

    def _build(self):
        title_key, sub_key = _META.get(
            self.rol_tipo, ("title_alumnos", "sub_alumnos"))
        c = tk.Frame(self, bg=CARD, padx=14, pady=12)
        c.pack(fill="both", expand=True, padx=18, pady=14)

        # Cabecera
        hdr = tk.Frame(c, bg=CARD)
        hdr.pack(fill="x", pady=(0, 10))
        card_head(hdr, t(title_key), t(sub_key))

        bframe = tk.Frame(hdr, bg=CARD)
        bframe.pack(side="right")
        tk.Button(bframe, text=t("editar"), command=self._editar_sel,
                  bg=BLUE, fg="white", relief="flat", font=("Arial", 9),
                  padx=10, pady=5, cursor="hand2").pack(side="left", padx=4)
        tk.Button(bframe, text=t("eliminar"), command=self._eliminar_sel,
                  bg=RED, fg="white", relief="flat", font=("Arial", 9),
                  padx=10, pady=5, cursor="hand2").pack(side="left", padx=4)
        tk.Button(bframe, text=t("agregar"), command=self._agregar,
                  bg=ACCENT, fg=BG, relief="flat",
                  font=("Arial", 9, "bold"),
                  padx=10, pady=5, cursor="hand2").pack(side="left", padx=4)

        # Búsqueda
        srow = tk.Frame(c, bg=CARD)
        srow.pack(fill="x", pady=(0, 8))
        tk.Label(srow, text=t("buscar_lbl"), bg=CARD, fg=T2,
                 font=("Arial", 9)).pack(side="left")
        self._sq = tk.StringVar()
        self._sq.trace_add("write", lambda *_: self._filtrar())
        tk.Entry(srow, textvariable=self._sq, width=30,
                 bg=CARD2, fg=T1, insertbackground=T1,
                 relief="flat", font=("Arial", 9)
                 ).pack(side="left", padx=6, ipady=4)
        self._lbl_n = tk.Label(srow, text="", bg=CARD, fg=T3,
                                font=("Arial", 8))
        self._lbl_n.pack(side="right")

        # Tabla
        if self.rol_tipo == "estudiante":
            cols = [
                ("nombre", t("col_nombre"),      230, "w"),
                ("cuenta", t("col_cuenta"),      110, "center"),
                ("grado",  t("col_grado_grupo"),  90, "center"),
                ("freg",   t("col_registrado"),  100, "center"),
                ("actpor", t("col_act_por"),      160, "w"),
                ("fact",   t("col_ult_act"),      100, "center"),
            ]
        else:
            cols = [
                ("nombre", t("col_nombre"),      220, "w"),
                ("cuenta", t("col_cuenta"),      110, "center"),
                ("correo", t("correo"),           200, "w"),
                ("freg",   t("col_registrado"),  100, "center"),
                ("actpor", t("col_act_por"),      160, "w"),
                ("fact",   t("col_ult_act"),      100, "center"),
            ]
        wrap = tk.Frame(c, bg=CARD)
        wrap.pack(fill="both", expand=True)
        self.tree = make_treeview(wrap, cols, height=22)

    def on_show(self):
        self.refresh()

    def refresh(self):
        self._datos = _listar(self.rol_tipo)
        self._filtrar()

    def _filtrar(self):
        q = self._sq.get().strip().lower()
        filt = ([d for d in self._datos
                 if q in (d["nombre"] + d["ap"] + d["cuenta"]).lower()]
                if q else self._datos)

        self.tree.delete(*self.tree.get_children())
        for d in filt:
            nombre_completo = f"{d['nombre']} {d['ap']} {d['am']}".strip()
            if self.rol_tipo == "estudiante":
                gg = (f"{d['grado']}° {d['grupo']}"
                      if d["grado"] else "—")
                self.tree.insert("", "end", iid=str(d["id"]), values=(
                    nombre_completo, d["cuenta"] or "—", gg,
                    d["fecha_reg"] or "—", d["act_por"], d["fecha_act"],
                ))
            else:
                self.tree.insert("", "end", iid=str(d["id"]), values=(
                    nombre_completo, d["cuenta"] or "—",
                    d["correo"] or "—",
                    d["fecha_reg"] or "—", d["act_por"], d["fecha_act"],
                ))

        n   = len(filt)
        key = "registros" if n == 1 else "registros_pl"
        self._lbl_n.configure(text=f"{n} {t(key)}")

    def _selected(self) -> dict | None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo(t("seleccion"), t("selecciona_reg"),
                                parent=self.dash.root)
            return None
        return next(
            (d for d in self._datos if d["id"] == int(sel[0])), None)

    def _agregar(self):
        if self.rol_tipo == "estudiante":
            EstudianteFormDialog(
                self.dash.root, self.dash, on_save=self.refresh)
        else:
            PersonFormDialog(
                self.dash.root, self.rol_tipo, self.dash,
                on_save=self.refresh)

    def _editar_sel(self):
        d = self._selected()
        if d:
            PersonFormDialog(
                self.dash.root, self.rol_tipo, self.dash,
                datos_existentes=d, on_save=self.refresh)

    def _eliminar_sel(self):
        d = self._selected()
        if not d:
            return
        if d["id"] == self.dash.usuario.get("id"):
            messagebox.showerror(t("error"), t("no_autoeliminar"),
                                 parent=self.dash.root)
            return
        DeleteDialog(
            self.dash.root,
            f"{d['nombre']} {d['ap']}",
            d["id"],
            on_confirm=self.refresh,
        )


# ═══════════════════════════════════════════════════════════════════
#  PERSON FORM DIALOG  (Maestros y Admins)
# ═══════════════════════════════════════════════════════════════════

class PersonFormDialog(tk.Toplevel):
    """
    Formulario de alta / edición para maestros y admins.
    Al crear: llama a CapturaBiometricaDialog cuando los datos se guardan.
    """

    def __init__(self, parent, rol_tipo: str, dashboard,
                 datos_existentes: dict | None = None,
                 on_save=None):
        super().__init__(parent)
        self.dashboard = dashboard
        self.rol_tipo  = rol_tipo
        self.datos_ex  = datos_existentes
        self.on_save   = on_save
        self._edit     = datos_existentes is not None

        accion  = "editar" if self._edit else "agregar"
        rol_key = {"estudiante": "alumno",
                   "maestro":   "maestro",
                   "admin":     "admin"}.get(rol_tipo, "admin")
        titulo  = t(f"{accion}_{rol_key}")

        self.title(titulo)
        self.configure(bg=CARD)
        self.resizable(False, False)
        self.transient(parent)
        self._build(titulo)
        if self._edit:
            self._prefill()
        self._finalize(parent)

    def _finalize(self, parent):
        self.update_idletasks()
        W_DLG = 480
        H_DLG = max(self.winfo_reqheight(), 300)
        px = parent.winfo_rootx() + parent.winfo_width()  // 2 - W_DLG // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - H_DLG // 2
        self.geometry(f"{W_DLG}x{H_DLG}+{px}+{py}")
        self.grab_set()
        self.lift()
        self.focus_force()

    def _build(self, titulo: str):
        tk.Label(self, text=titulo, bg=CARD, fg=T1,
                 font=("Arial", 13, "bold")).pack(padx=22, pady=(18, 4))
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=22, pady=2)

        self._v_nombre  = tk.StringVar()
        self._v_ap      = tk.StringVar()
        self._v_am      = tk.StringVar()
        self._v_cuenta  = tk.StringVar()
        self._v_correo  = tk.StringVar()
        self._v_pwd     = tk.StringVar()
        self._v_err     = tk.StringVar()

        form = tk.Frame(self, bg=CARD)
        form.pack(fill="x", padx=22, pady=4)

        self._lf(form, t("nombres"), self._v_nombre)

        rowa = tk.Frame(form, bg=CARD)
        rowa.pack(fill="x", pady=3)
        c1 = tk.Frame(rowa, bg=CARD)
        c1.pack(side="left", fill="x", expand=True, padx=(0, 6))
        c2 = tk.Frame(rowa, bg=CARD)
        c2.pack(side="left", fill="x", expand=True)
        self._lf(c1, t("ap_pat"), self._v_ap)
        self._lf(c2, t("ap_mat"), self._v_am)

        self._lf(form, t("cuenta_campo"), self._v_cuenta)

        if self.rol_tipo in ("maestro", "admin"):
            self._lf(form, t("correo_campo"), self._v_correo)
            pwd_lbl = t("pwd_nuevo") if not self._edit else t("pwd_editar")
            self._lf(form, pwd_lbl, self._v_pwd, show="*")

        # Aviso de captura facial solo al crear
        if not self._edit:
            info = tk.Frame(form, bg="#1a1f2e", padx=10, pady=8)
            info.pack(fill="x", pady=(15, 5))
            tk.Label(
                info,
                text="📷  Después de guardar se abrirá la captura\n"
                     "biométrica (OBLIGATORIO).",
                bg="#1a1f2e", fg="#8ab4f8",
                font=("Arial", 8), justify="left",
            ).pack()

        tk.Label(self, textvariable=self._v_err, bg=CARD, fg=RED,
                 font=("Arial", 8), wraplength=400).pack(padx=22)

        tk.Frame(self, bg=BORDER, height=1).pack(
            fill="x", padx=22, pady=(6, 0))
        brow = tk.Frame(self, bg=CARD)
        brow.pack(fill="x", padx=22, pady=(8, 18))
        tk.Button(brow, text=t("cancelar"), command=self.destroy,
                  bg=CARD2, fg=T2, relief="flat", font=("Arial", 9),
                  padx=12, pady=6, cursor="hand2").pack(side="left")
        tk.Button(brow, text=t("guardar"), command=self._save,
                  bg=ACCENT, fg=BG, relief="flat",
                  font=("Arial", 9, "bold"),
                  padx=16, pady=6, cursor="hand2").pack(side="right")

    def _lf(self, parent, label: str, var: tk.StringVar, show: str = ""):
        tk.Label(parent, text=label, bg=CARD, fg=T2,
                 font=("Arial", 8), anchor="w").pack(fill="x")
        kw = dict(textvariable=var, bg=CARD2, fg=T1,
                  insertbackground=T1, relief="flat",
                  font=("Arial", 10))
        if show:
            kw["show"] = show
        e = tk.Entry(parent, **kw)
        e.pack(fill="x", ipady=5, pady=(0, 4))
        return e

    def _prefill(self):
        d = self.datos_ex
        self._v_nombre.set(d.get("nombre", ""))
        self._v_ap.set(d.get("ap", ""))
        self._v_am.set(d.get("am", ""))
        self._v_cuenta.set(d.get("cuenta", ""))
        self._v_correo.set(d.get("correo", ""))

    def _save(self):
        self._v_err.set("")
        nombre = self._v_nombre.get().strip()
        ap     = self._v_ap.get().strip()
        am     = self._v_am.get().strip()
        cuenta = self._v_cuenta.get().strip()
        correo = self._v_correo.get().strip()
        pwd    = self._v_pwd.get()

        if not nombre or not ap:
            self._v_err.set(t("err_nombre")); return
        if not cuenta or not cuenta.isdigit() or len(cuenta) != 8:
            self._v_err.set(t("err_cuenta")); return
        if self.rol_tipo in ("maestro", "admin") and not correo:
            self._v_err.set(t("err_correo")); return
        if self.rol_tipo in ("maestro", "admin") and not self._edit and not pwd:
            self._v_err.set(t("err_pwd")); return

        datos = {
            "id":         self.datos_ex["id"] if self._edit else None,
            "nombre":     nombre,
            "ap":         ap,
            "am":         am,
            "cuenta":     cuenta,
            "correo":     correo or None,
            "contrasena": pwd or None,
            "rol":        self.rol_tipo,
        }
        editor = self.dashboard.usuario.get("id", 0)
        ok, msg = (_editar(datos, editor) if self._edit
                   else _crear(datos, editor))

        if ok:
            if self._edit:
                messagebox.showinfo(t("exito"), msg, parent=self)
                if self.on_save:
                    self.on_save()
                self.destroy()
            else:
                # ── Captura biométrica OBLIGATORIA ──────────────────
                self.destroy()
                self._abrir_captura(cuenta, nombre, ap, am,
                                     correo, self.rol_tipo)
        else:
            self._v_err.set(f"✗ {msg}")

    def _abrir_captura(self, cuenta, nombre, ap, am, correo, rol):
        """Busca el uid recién creado y abre CapturaBiometricaDialog."""
        try:
            conn = _db()
            c    = conn.cursor()
            c.execute(
                "SELECT id FROM usuarios WHERE numero_cuenta=? AND rol=?",
                (cuenta, rol))
            row = c.fetchone()
            conn.close()
            uid = row[0] if row else None
        except Exception as e:
            print(f"[GESTION] No se pudo obtener uid: {e}")
            uid = None

        if uid is None:
            messagebox.showerror(
                t("error"),
                "No se encontró el registro recién creado.",
                parent=self.dashboard.root)
            return

        from registro_facial_dash import CapturaBiometricaDialog
        datos = {
            "nombre": nombre, "ap": ap, "am": am,
            "cuenta": cuenta, "rol": rol, "correo": correo,
        }
        CapturaBiometricaDialog(
            parent     = self.dashboard.root,
            uid        = uid,
            datos      = datos,
            on_success = self.on_save,
            on_cancel  = self.on_save,
        )


# ═══════════════════════════════════════════════════════════════════
#  ESTUDIANTE FORM DIALOG
# ═══════════════════════════════════════════════════════════════════

class EstudianteFormDialog(tk.Toplevel):
    """
    Formulario para registrar un estudiante + captura biométrica integrada.
    """

    def __init__(self, parent, dashboard, on_save=None):
        super().__init__(parent)
        self.dashboard = dashboard
        self.on_save   = on_save

        self.title(t("agregar_alumno"))
        self.configure(bg=CARD)
        self.resizable(False, False)
        self.transient(parent)

        self._build()
        self._finalize(parent)

    def _finalize(self, parent):
        self.update_idletasks()
        W_DLG = 480
        H_DLG = max(self.winfo_reqheight(), 420)
        px = parent.winfo_rootx() + parent.winfo_width()  // 2 - W_DLG // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - H_DLG // 2
        self.geometry(f"{W_DLG}x{H_DLG}+{px}+{py}")
        self.grab_set()
        self.lift()
        self.focus_force()

    def _build(self):
        tk.Label(self, text=t("agregar_alumno"), bg=CARD, fg=T1,
                 font=("Arial", 13, "bold")).pack(padx=22, pady=(18, 4))
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=22, pady=2)

        self._v_nombre = tk.StringVar()
        self._v_ap     = tk.StringVar()
        self._v_am     = tk.StringVar()
        self._v_cuenta = tk.StringVar()
        self._v_grado  = tk.StringVar()
        self._v_grupo  = tk.StringVar()
        self._v_err    = tk.StringVar()

        form = tk.Frame(self, bg=CARD)
        form.pack(fill="x", padx=22, pady=4)

        self._campo(form, t("nombres"), self._v_nombre)

        row_ap = tk.Frame(form, bg=CARD)
        row_ap.pack(fill="x", pady=3)
        c1 = tk.Frame(row_ap, bg=CARD)
        c1.pack(side="left", fill="x", expand=True, padx=(0, 6))
        c2 = tk.Frame(row_ap, bg=CARD)
        c2.pack(side="left", fill="x", expand=True)
        self._campo(c1, t("ap_pat"), self._v_ap)
        self._campo(c2, t("ap_mat"), self._v_am)

        self._campo(form, t("cuenta_campo"), self._v_cuenta)

        row_gg = tk.Frame(form, bg=CARD)
        row_gg.pack(fill="x", pady=3)
        g1 = tk.Frame(row_gg, bg=CARD)
        g1.pack(side="left", fill="x", expand=True, padx=(0, 6))
        g2 = tk.Frame(row_gg, bg=CARD)
        g2.pack(side="left", fill="x", expand=True)
        self._campo(g1, t("grado_campo"), self._v_grado)
        self._campo(g2, t("grupo_campo"), self._v_grupo)

        # Aviso de captura facial
        info = tk.Frame(form, bg="#1a1f2e", padx=10, pady=8)
        info.pack(fill="x", pady=(15, 5))
        tk.Label(
            info,
            text="📷  Después de guardar se abrirá la captura\n"
                 "biométrica (OBLIGATORIO).",
            bg="#1a1f2e", fg="#8ab4f8",
            font=("Arial", 8), justify="left",
        ).pack()

        tk.Label(self, textvariable=self._v_err, bg=CARD, fg=RED,
                 font=("Arial", 8), wraplength=400).pack(padx=22)

        tk.Frame(self, bg=BORDER, height=1).pack(
            fill="x", padx=22, pady=(6, 0))
        brow = tk.Frame(self, bg=CARD)
        brow.pack(fill="x", padx=22, pady=(8, 18))

        tk.Button(brow, text=t("cancelar"), command=self.destroy,
                  bg=CARD2, fg=T2, relief="flat", font=("Arial", 9),
                  padx=12, pady=6, cursor="hand2").pack(side="left")
        tk.Button(
            brow, text="Continuar al Registro Facial →",
            command=self._continuar,
            bg=ACCENT, fg=BG, relief="flat",
            font=("Arial", 9, "bold"),
            padx=16, pady=6, cursor="hand2",
        ).pack(side="right")

    def _campo(self, parent, label: str, var: tk.StringVar):
        tk.Label(parent, text=label, bg=CARD, fg=T2,
                 font=("Arial", 8), anchor="w").pack(fill="x")
        e = tk.Entry(parent, textvariable=var, bg=CARD2, fg=T1,
                     insertbackground=T1, relief="flat",
                     font=("Arial", 10))
        e.pack(fill="x", ipady=5, pady=(0, 4))
        return e

    def _continuar(self):
        self._v_err.set("")

        nombre = self._v_nombre.get().strip()
        ap     = self._v_ap.get().strip()
        am     = self._v_am.get().strip()
        cuenta = self._v_cuenta.get().strip()
        grado  = self._v_grado.get().strip()
        grupo  = self._v_grupo.get().strip().upper()

        if not nombre or not ap:
            self._v_err.set(t("err_nombre")); return
        if not cuenta or not cuenta.isdigit() or len(cuenta) != 8:
            self._v_err.set(t("err_cuenta")); return
        if not grado or not grado.isdigit():
            self._v_err.set(t("err_grado")); return
        if not grupo or len(grupo) != 1 or not grupo.isalpha():
            self._v_err.set(t("err_grupo")); return

        datos_bd = {
            "nombre": nombre, "ap": ap, "am": am,
            "cuenta": cuenta,
            "grado":  int(grado), "grupo": grupo,
            "rol":    "estudiante",
        }
        editor_id = self.dashboard.usuario.get("id", 0)
        ok, msg   = _crear(datos_bd, editor_id)

        if not ok:
            self._v_err.set(f"✗ {msg}")
            return

        # Obtener uid recién creado
        try:
            conn = _db()
            c    = conn.cursor()
            c.execute(
                "SELECT id FROM usuarios "
                "WHERE numero_cuenta=? AND rol='estudiante'",
                (cuenta,))
            row = c.fetchone()
            conn.close()
            uid = row[0] if row else None
        except Exception as e:
            print(f"[GESTION] No se pudo obtener uid: {e}")
            uid = None

        if uid is None:
            messagebox.showerror(
                t("error"),
                "No se encontró el registro recién creado.",
                parent=self.dashboard.root)
            return

        self.destroy()

        from registro_facial_dash import CapturaBiometricaDialog
        datos_disp = {
            "nombre": nombre, "ap": ap, "am": am,
            "cuenta": cuenta, "rol": "estudiante",
            "grado":  grado,  "grupo": grupo,
        }
        CapturaBiometricaDialog(
            parent     = self.dashboard.root,
            uid        = uid,
            datos      = datos_disp,
            on_success = self.on_save,
            on_cancel  = self.on_save,
        )


# ═══════════════════════════════════════════════════════════════════
#  DELETE DIALOG
# ═══════════════════════════════════════════════════════════════════

class DeleteDialog(tk.Toplevel):
    def __init__(self, parent, nombre: str, uid: int, on_confirm=None):
        super().__init__(parent)
        self.uid        = uid
        self.on_confirm = on_confirm
        self.title(t("eliminar_titulo"))
        self.configure(bg=CARD)
        self.resizable(False, False)
        self.transient(parent)

        tk.Label(self, text=f"🗑  {t('eliminar_titulo')}", bg=CARD, fg=T1,
                 font=("Arial", 13, "bold")).pack(padx=20, pady=(22, 8))
        tk.Label(
            self,
            text=t("eliminar_pregunta", nombre=nombre),
            bg=CARD, fg=T2, font=("Arial", 10),
            justify="center", wraplength=320,
        ).pack(padx=20)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20, pady=12)
        brow = tk.Frame(self, bg=CARD)
        brow.pack(padx=20, pady=(0, 20))
        tk.Button(brow, text=t("cancelar"), command=self.destroy,
                  bg=CARD2, fg=T2, relief="flat", font=("Arial", 9),
                  padx=12, pady=6, cursor="hand2").pack(side="left", padx=6)
        tk.Button(brow, text=t("eliminar"), command=self._confirm,
                  bg=RED, fg="white", relief="flat",
                  font=("Arial", 9, "bold"),
                  padx=16, pady=6, cursor="hand2").pack(side="left", padx=6)

        W_DLG, H_DLG = 380, 220
        px = parent.winfo_rootx() + parent.winfo_width()  // 2 - W_DLG // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - H_DLG // 2
        self.geometry(f"{W_DLG}x{H_DLG}+{px}+{py}")
        self.grab_set()
        self.lift()
        self.focus_force()

    def _confirm(self):
        ok, msg = _eliminar(self.uid)
        if ok:
            messagebox.showinfo(t("exito"), msg, parent=self)
            if self.on_confirm:
                self.on_confirm()
            self.destroy()
        else:
            messagebox.showerror(t("error"), msg, parent=self)