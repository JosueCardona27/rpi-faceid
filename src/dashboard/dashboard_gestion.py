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
from lang_dict import t, toggle_lang, fecha_local

# ═══════════════════════════════════════════════════════════════════
#  TECLADO VIRTUAL / AVISOS TÁCTILES 7"
# ═══════════════════════════════════════════════════════════════════

def _center_on_parent(win, parent, width: int, height: int):
    """
    Centra una ventana Tk sobre el dashboard.

    En Windows/Raspbian a veces el parent reporta 1x1 o coordenadas raras
    cuando el dashboard está maximizado/kiosko. Por eso se usa primero el
    toplevel real y, si no hay medidas confiables, se centra en la pantalla.
    """
    try:
        root = parent.winfo_toplevel() if parent is not None else None
        base = root or parent
        if base is not None:
            base.update_idletasks()

        sw = base.winfo_screenwidth()  if base is not None else win.winfo_screenwidth()
        sh = base.winfo_screenheight() if base is not None else win.winfo_screenheight()

        px = py = None
        if root is not None:
            rw = root.winfo_width()
            rh = root.winfo_height()
            rx = root.winfo_rootx()
            ry = root.winfo_rooty()
            # Si el toplevel ya está dibujado, centrar respecto al dashboard.
            if rw > 80 and rh > 80:
                px = rx + (rw - width)  // 2
                py = ry + (rh - height) // 2

        # Fallback: centro real de pantalla.
        if px is None or py is None:
            px = (sw - width)  // 2
            py = (sh - height) // 2

        # Evita que la ventana se quede fuera de pantalla por bordes negativos
        # de ventanas maximizadas o por el modo kiosko en Raspberry.
        px = max(0, min(int(px), max(0, sw - width)))
        py = max(0, min(int(py), max(0, sh - height)))
    except Exception:
        try:
            sw = win.winfo_screenwidth()
            sh = win.winfo_screenheight()
            px = max(0, (sw - width)  // 2)
            py = max(0, (sh - height) // 2)
        except Exception:
            px, py = 20, 20

    win.geometry(f"{width}x{height}+{px}+{py}")
    try:
        win.update_idletasks()
        win.lift(parent)
        win.focus_force()
    except Exception:
        pass


def _touch_mode_from_dashboard(dashboard) -> bool:
    return bool(
        getattr(dashboard, "rpi_mode", False)
        or getattr(dashboard, "portrait_mode", False)
        or getattr(dashboard, "compact_mode", False)
    )


# Ventanas táctiles abiertas por gestión.
# Sirve para cerrar teclado/formularios cuando el usuario cambia de pestaña.
_OPEN_TOUCH_WINDOWS: set = set()
_ACTIVE_KEYBOARD = None


def _register_touch_window(win):
    try:
        _OPEN_TOUCH_WINDOWS.add(win)
        win.bind(
            "<Destroy>",
            lambda _e, w=win: _OPEN_TOUCH_WINDOWS.discard(w),
            add="+",
        )
    except Exception:
        pass


def _close_touch_windows(except_win=None):
    global _ACTIVE_KEYBOARD
    for w in list(_OPEN_TOUCH_WINDOWS):
        if except_win is not None and w is except_win:
            continue
        try:
            if w.winfo_exists():
                w.grab_release()
                w.destroy()
        except Exception:
            pass
    try:
        if _ACTIVE_KEYBOARD is not None and not _ACTIVE_KEYBOARD.winfo_exists():
            _ACTIVE_KEYBOARD = None
    except Exception:
        _ACTIVE_KEYBOARD = None


def _install_navigation_cleanup(dashboard):
    """Cierra teclado/formularios al cambiar de pestaña sin tocar dashboard.py."""
    if getattr(dashboard, "_gestion_cleanup_installed", False):
        return
    original_nav = getattr(dashboard, "navigate", None)
    if not callable(original_nav):
        return

    def _navigate_with_cleanup(view_id: str, *args, **kwargs):
        actual = getattr(dashboard, "_current", "")
        if view_id != actual:
            _close_touch_windows()
        return original_nav(view_id, *args, **kwargs)

    dashboard.navigate = _navigate_with_cleanup
    dashboard._gestion_cleanup_installed = True


class TouchKeyboardDialog(tk.Toplevel):
    """
    Teclado virtual dibujado 100% en Tkinter.
    No usa diálogos nativos del sistema operativo, para evitar bloqueos en Raspberry.
    """

    def __init__(self, parent, title: str, initial: str = "", on_accept=None,
                 show: str = "", accept_text: str = "Aceptar",
                 submit_text: str = "", on_submit=None):
        super().__init__(parent)
        global _ACTIVE_KEYBOARD
        try:
            if _ACTIVE_KEYBOARD is not None and _ACTIVE_KEYBOARD.winfo_exists():
                _ACTIVE_KEYBOARD.grab_release()
                _ACTIVE_KEYBOARD.destroy()
        except Exception:
            pass
        _ACTIVE_KEYBOARD = self
        _register_touch_window(self)

        self.parent = parent
        self.on_accept = on_accept
        self.on_submit = on_submit
        self.accept_text = accept_text or "Aceptar"
        self.submit_text = submit_text or ""
        self.show_char = show or ""
        try:
            self._previous_grab = parent.grab_current()
        except Exception:
            self._previous_grab = None
        self._shift = False
        self._value = tk.StringVar(value=initial or "")

        self.configure(bg="#0F172A")
        self.resizable(False, False)
        self.transient(parent)
        try:
            self.overrideredirect(True)
        except Exception:
            pass

        sw = max(480, parent.winfo_screenwidth())
        sh = max(480, parent.winfo_screenheight())

        # Responsive para Raspberry Pi OS / Raspbian:
        # - 600x1024 vertical: casi todo el ancho, con botones visibles.
        # - 1024x600 horizontal: ancho completo útil y altura dentro de pantalla.
        # - Escritorio: tamaño cómodo, centrado sobre el dashboard.
        self._small_touch = (sw <= 700 or sh <= 650)
        if sw <= 700 and sh > sw:
            width  = max(520, min(sw - 16, 584))
            height = max(390, min(480, sh - 32))
        elif sh <= 650:
            width  = max(720, min(940, sw - 16))
            height = max(390, min(500, sh - 20))
        else:
            width  = min(720, sw - 24)
            height = min(440, sh - 70)
        _center_on_parent(self, parent, int(width), int(height))

        outer = tk.Frame(self, bg="#0F172A", padx=2, pady=2)
        outer.pack(fill="both", expand=True)
        box = tk.Frame(outer, bg="#F8FAFC")
        box.pack(fill="both", expand=True)

        header = tk.Frame(box, bg=BLUE, height=46)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text=f"⌨  {title}", bg=BLUE, fg="#FFFFFF",
                 font=("Segoe UI", 11 if self._small_touch else 12, "bold"), anchor="w").pack(
                     side="left", fill="both", expand=True, padx=14)
        tk.Button(header, text="✕", command=self._cancel,
                  bg=BLUE, fg="#FFFFFF", activebackground=RED,
                  activeforeground="#FFFFFF", relief="flat", bd=0,
                  font=("Segoe UI", 12, "bold"), cursor="hand2").pack(
                      side="right", padx=6, ipadx=8, ipady=6)

        input_frame = tk.Frame(box, bg="#F8FAFC")
        input_frame.pack(fill="x", padx=12, pady=(10, 6))
        self.entry = tk.Entry(input_frame, textvariable=self._value,
                              show=self.show_char,
                              bg="#FFFFFF", fg=T1, insertbackground=ACCENT,
                              relief="flat", font=("Segoe UI", 14 if self._small_touch else 16))
        self.entry.pack(fill="x", ipady=8)
        self.entry.focus_set()
        try:
            self.entry.icursor(tk.END)
        except Exception:
            pass
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<KP_Enter>", self._on_return)
        self.entry.bind("<Escape>", lambda _e: (self._cancel(), "break")[1])
        self.bind("<Return>", self._on_return)
        self.bind("<KP_Enter>", self._on_return)
        self.bind("<Escape>", lambda _e: (self._cancel(), "break")[1])

        self.keys_holder = tk.Frame(box, bg="#F8FAFC")
        self.keys_holder.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self._draw_keys()

        footer = tk.Frame(box, bg="#F8FAFC")
        footer.pack(fill="x", padx=10, pady=(0, 10))
        tk.Button(footer, text="Cancelar", command=self._cancel,
                  bg=CARD2, fg=T2, activebackground="#D7D1C4",
                  relief="flat", bd=0,
                  font=("Segoe UI", 9 if self._small_touch else 10, "bold"),
                  cursor="hand2", padx=14 if self._small_touch else 18,
                  pady=7 if self._small_touch else 8).pack(side="left")

        actions = tk.Frame(footer, bg="#F8FAFC")
        actions.pack(side="right")

        # Aceptar solo aplica el texto al campo y cierra el teclado.
        # El guardado real se hace con el botón Guardar del formulario de usuario.
        tk.Button(actions, text=self.accept_text, command=self._accept,
                  bg=BLUE, fg="#FFFFFF", activebackground="#2E4068",
                  activeforeground="#FFFFFF", relief="flat", bd=0,
                  font=("Segoe UI", 9 if self._small_touch else 10, "bold"), cursor="hand2",
                  padx=14 if self._small_touch else 22, pady=7 if self._small_touch else 8).pack(side="left")

        try:
            self.grab_set()
            self.lift()
            self.focus_force()
        except Exception:
            pass

    def _on_return(self, _event=None):
        # Enter dentro del teclado virtual solo aplica el texto al campo.
        # Para guardar cambios se usa el botón Guardar del formulario.
        self._accept()
        return "break"

    def _restore_parent_grab(self):
        try:
            if self._previous_grab is not None and self._previous_grab.winfo_exists():
                self._previous_grab.grab_set()
                self._previous_grab.focus_force()
            elif self.parent is not None and self.parent.winfo_exists():
                self.parent.focus_force()
        except Exception:
            pass

    def _cancel(self):
        global _ACTIVE_KEYBOARD
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            _OPEN_TOUCH_WINDOWS.discard(self)
            if _ACTIVE_KEYBOARD is self:
                _ACTIVE_KEYBOARD = None
            self.destroy()
        finally:
            self._restore_parent_grab()

    def _accept(self):
        value = self._value.get()
        try:
            if self.on_accept:
                self.on_accept(value)
        finally:
            self._cancel()

    def _submit(self):
        """Aplica el texto del teclado y ejecuta la acción principal del formulario."""
        value = self._value.get()
        cb = self.on_submit
        try:
            if self.on_accept:
                self.on_accept(value)
        finally:
            global _ACTIVE_KEYBOARD
            try:
                self.grab_release()
            except Exception:
                pass
            try:
                _OPEN_TOUCH_WINDOWS.discard(self)
                if _ACTIVE_KEYBOARD is self:
                    _ACTIVE_KEYBOARD = None
                self.destroy()
            finally:
                self._restore_parent_grab()
        if cb:
            try:
                if self.parent is not None and self.parent.winfo_exists():
                    self.parent.after(40, cb)
                else:
                    cb()
            except Exception as e:
                print(f"[TECLADO] Error al guardar desde teclado: {e}")

    def _insert(self, txt: str):
        try:
            pos = self.entry.index(tk.INSERT)
            self.entry.insert(pos, txt)
            self.entry.icursor(pos + len(txt))
        except Exception:
            self._value.set(self._value.get() + txt)

    def _backspace(self):
        try:
            pos = self.entry.index(tk.INSERT)
            if pos > 0:
                self.entry.delete(pos - 1, pos)
        except Exception:
            self._value.set(self._value.get()[:-1])

    def _clear(self):
        self._value.set("")

    def _press(self, key: str):
        if key == "__SUBMIT__":
            self._submit()
        elif key == "__ACCEPT__":
            self._accept()
        elif key == "__CANCEL__":
            self._cancel()
        elif key == "⌫":
            self._backspace()
        elif key == "Limpiar":
            self._clear()
        elif key == "Espacio":
            self._insert(" ")
        elif key == "⇧":
            self._shift = not self._shift
            self._draw_keys()
        else:
            if len(key) == 1 and key.isalpha():
                self._insert(key.upper() if self._shift else key.lower())
            else:
                self._insert(key)

    def _button(self, parent, text, bg="#FFFFFF", fg=T1, wide=False, value=None):
        if len(text) == 1 and text.isalpha():
            label = text.upper() if self._shift else text.lower()
        else:
            label = text
        press_value = value if value is not None else text
        b = tk.Button(parent, text=label, command=lambda k=press_value: self._press(k),
                      bg=bg, fg=fg, activebackground="#DDF2EA",
                      activeforeground=T1 if bg != ACCENT else "#FFFFFF", relief="flat", bd=0,
                      highlightthickness=1, highlightbackground="#D8DEE9",
                      font=("Segoe UI", 9 if getattr(self, "_small_touch", False) else 10, "bold"), cursor="hand2")
        b.pack(side="left", fill="both", expand=True, padx=2, pady=2,
               ipadx=10 if wide else 2)
        return b

    def _draw_keys(self):
        for child in self.keys_holder.winfo_children():
            child.destroy()
        layouts = [
            list("1234567890"),
            list("qwertyuiop"),
            list("asdfghjklñ"),
            ["⇧"] + list("zxcvbnm") + ["⌫"],
            ["@", "_", "-", ".", "/", "Espacio", ".com", "Limpiar"],
        ]
        for row_keys in layouts:
            row = tk.Frame(self.keys_holder, bg="#F8FAFC")
            row.pack(fill="both", expand=True)
            for key in row_keys:
                if key == "⇧":
                    self._button(row, key,
                                 bg=ACCENT if self._shift else "#FFFFFF",
                                 fg="#FFFFFF" if self._shift else T1)
                elif key in ("⌫", "Limpiar"):
                    self._button(row, key, bg="#FFECEC", fg=RED,
                                 wide=(key == "Limpiar"))
                elif key in ("Espacio", ".com"):
                    self._button(row, key, bg="#EAF7F0", fg=ACCENT,
                                 wide=True)
                else:
                    self._button(row, key)



def _abrir_teclado(parent, title: str, var: tk.StringVar, show: str = "",
                   accept_text: str = "Aceptar", on_accept=None,
                   submit_text: str = "", on_submit=None):
    """Abre teclado táctil y escribe el resultado en el StringVar indicado."""
    def _ok(value):
        var.set(value)
        if on_accept:
            on_accept(value)
    TouchKeyboardDialog(parent, title=title, initial=var.get(),
                        on_accept=_ok, show=show, accept_text=accept_text,
                        submit_text=submit_text, on_submit=on_submit)


def _attach_virtual_keyboard(entry: tk.Entry, parent, label: str,
                             var: tk.StringVar, show: str = "",
                             accept_text: str = "Aceptar", on_accept=None,
                             submit_text: str = "", on_submit=None):
    """Hace que un Entry abra teclado virtual al tocarlo; Enter físico puede guardar."""
    def _open(_event=None):
        try:
            if getattr(entry, "_vk_opening", False):
                return "break"
            entry._vk_opening = True
        except Exception:
            pass

        def _launch():
            try:
                _abrir_teclado(parent, label, var, show=show,
                               accept_text=accept_text, on_accept=on_accept,
                               submit_text=submit_text, on_submit=on_submit)
            finally:
                try:
                    entry.after(450, lambda: setattr(entry, "_vk_opening", False))
                except Exception:
                    try:
                        entry._vk_opening = False
                    except Exception:
                        pass

        try:
            entry.after(35, _launch)
        except Exception:
            _launch()
        return "break"

    def _return(_event=None):
        if on_submit:
            on_submit()
        elif on_accept:
            on_accept(var.get())
        return "break"

    entry.configure(cursor="hand2")
    entry.bind("<Button-1>", _open)
    entry.bind("<ButtonRelease-1>", _open, add="+")
    entry.bind("<Double-Button-1>", _open)
    entry.bind("<FocusIn>", lambda e: entry.after(120, _open), add="+")
    entry.bind("<Return>", _return)
    entry.bind("<KP_Enter>", _return)
    return entry


def _touch_notice(parent, title: str, message: str, kind: str = "info",
                  auto_close_ms: int | None = 1400):
    """Aviso no nativo para Raspberry; reemplaza messagebox en gestión."""
    try:
        dlg = tk.Toplevel(parent)
        dlg.configure(bg="#0F172A")
        dlg.resizable(False, False)
        dlg.transient(parent)
        try:
            dlg.overrideredirect(True)
        except Exception:
            pass
        color = {"ok": ACCENT, "success": ACCENT,
                 "error": RED, "warning": "#E07A00"}.get(kind, BLUE)
        sw = max(480, parent.winfo_screenwidth())
        sh = max(480, parent.winfo_screenheight())
        W_DLG = min(380, sw - 24)
        H_DLG = min(160, sh - 40)
        _center_on_parent(dlg, parent, W_DLG, H_DLG)
        outer = tk.Frame(dlg, bg="#0F172A", padx=2, pady=2)
        outer.pack(fill="both", expand=True)
        box = tk.Frame(outer, bg="#FFFFFF")
        box.pack(fill="both", expand=True)
        tk.Frame(box, bg=color, height=5).pack(fill="x")
        tk.Label(box, text=title, bg="#FFFFFF", fg=color,
                 font=("Segoe UI", 12, "bold")).pack(pady=(14, 4))
        tk.Label(box, text=message, bg="#FFFFFF", fg=T2,
                 font=("Segoe UI", 10), justify="center",
                 wraplength=W_DLG - 36).pack(padx=18, expand=True)
        tk.Button(box, text="Aceptar", command=dlg.destroy,
                  bg=color, fg="#FFFFFF", relief="flat", bd=0,
                  font=("Segoe UI", 9, "bold"), cursor="hand2",
                  padx=16, pady=5).pack(pady=(4, 12))
        try:
            dlg.grab_set()
        except Exception:
            pass
        dlg.lift()
        try:
            dlg.attributes("-topmost", True)
            dlg.after(180, lambda: dlg.winfo_exists() and dlg.attributes("-topmost", False))
        except Exception:
            pass
        if auto_close_ms:
            dlg.after(auto_close_ms, lambda: dlg.winfo_exists() and dlg.destroy())
    except Exception:
        print(f"[{title}] {message}")


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
        _install_navigation_cleanup(dashboard)
        self.dash         = dashboard
        self.rol_tipo     = rol_tipo
        self.compact_mode = getattr(dashboard, "compact_mode", False)
        self._datos:  list[dict] = []
        self._build()

    def _build(self):
        cp = self.compact_mode
        title_key, sub_key = _META.get(
            self.rol_tipo, ("title_alumnos", "sub_alumnos"))

        padx_card = 10 if cp else 14
        pady_card = 8  if cp else 12
        c = tk.Frame(self, bg="#FFFFFF", padx=padx_card, pady=pady_card)
        c.pack(fill="both", expand=True,
               padx=10 if cp else 18, pady=10 if cp else 14)

        # Cabecera
        hdr = tk.Frame(c, bg="#FFFFFF")
        hdr.pack(fill="x", pady=(0, 8))
        card_head(hdr, t(title_key), t(sub_key), compact=cp)

        bframe = tk.Frame(hdr, bg="#FFFFFF")
        bframe.pack(side="right")
        btn_px = 7 if cp else 10
        btn_py = 3 if cp else 5
        btn_fz = 8 if cp else 9
        tk.Button(bframe, text=t("editar"), command=self._editar_sel,
                  bg=BLUE, fg="white", relief="flat", font=("Segoe UI", btn_fz, "bold"),
                  padx=btn_px + 2, pady=btn_py, cursor="hand2", bd=0).pack(side="left", padx=3)
        tk.Button(bframe, text=t("eliminar"), command=self._eliminar_sel,
                  bg=RED, fg="white", relief="flat", font=("Segoe UI", btn_fz, "bold"),
                  padx=btn_px + 2, pady=btn_py, cursor="hand2", bd=0).pack(side="left", padx=3)
        # BOTÓN AGREGAR — comentado temporalmente (sin función activa)
        # Para reactivar, descomenta las siguientes líneas:
        # tk.Button(bframe, text=t("agregar"), command=self._agregar,
        #           bg=ACCENT, fg=BG, relief="flat",
        #           font=("Arial", btn_fz, "bold"),
        #           padx=btn_px, pady=btn_py, cursor="hand2").pack(side="left", padx=3)

        # Búsqueda
        srow = tk.Frame(c, bg="#FFFFFF")
        srow.pack(fill="x", pady=(0, 6))
        tk.Label(srow, text=t("buscar_lbl"), bg="#FFFFFF", fg=T2,
                 font=("Arial", 8 if cp else 9)).pack(side="left")
        self._sq = tk.StringVar()
        # La búsqueda en pantalla táctil se aplica al aceptar el teclado virtual.
        # Si se usa teclado físico, el filtro sigue funcionando en vivo.
        self._sq.trace_add("write", lambda *_: self._filtrar())
        self._search_entry = tk.Entry(
            srow, textvariable=self._sq,
            width=28 if cp else 34,
            bg="#F5F0E8", fg=T1, insertbackground=T1,
            relief="flat", font=("Arial", 9 if cp else 10))
        # En modo táctil/responsivo se expande para que sea fácil presionarla.
        self._search_entry.pack(side="left", padx=5, ipady=5 if cp else 4,
                                fill="x", expand=True)
        self._search_entry.bind("<Button-1>", self._abrir_busqueda_tactil)
        self._search_entry.bind("<ButtonRelease-1>", self._abrir_busqueda_tactil, add="+")
        self._search_entry.bind("<Double-Button-1>", self._abrir_busqueda_tactil)
        self._search_entry.bind("<FocusIn>", lambda e: self._search_entry.after(120, self._abrir_busqueda_tactil), add="+")
        self._search_entry.bind("<Return>", lambda e: (self._filtrar(), "break")[1])
        self._search_entry.bind("<KP_Enter>", lambda e: (self._filtrar(), "break")[1])
        self._lbl_n = tk.Label(srow, text="", bg="#FFFFFF", fg=T3,
                                font=("Arial", 7 if cp else 8))
        self._lbl_n.pack(side="right")

        # ── Tabla con scroll horizontal en compact ───────────────────────
        # estudiante: 190+95+80+90+140+90 = 685px → xscroll en compact
        # otros:      190+95+180+90+140+90 = 785px → xscroll en compact
        if self.rol_tipo == "estudiante":
            cols = [
                ("nombre", t("col_nombre"),      190, "w"),
                ("cuenta", t("col_cuenta"),       95, "center"),
                ("grado",  t("col_grado_grupo"),  80, "center"),
                ("freg",   t("col_registrado"),   90, "center"),
                ("actpor", t("col_act_por"),      140, "w"),
                ("fact",   t("col_ult_act"),       90, "center"),
            ]
        else:
            cols = [
                ("nombre", t("col_nombre"),      190, "w"),
                ("cuenta", t("col_cuenta"),       95, "center"),
                ("correo", t("correo"),           180, "w"),
                ("freg",   t("col_registrado"),   90, "center"),
                ("actpor", t("col_act_por"),      140, "w"),
                ("fact",   t("col_ult_act"),       90, "center"),
            ]

        wrap = tk.Frame(c, bg="#FFFFFF")
        wrap.pack(fill="both", expand=True)
        # En compact: xscroll=True para poder deslizar y ver todas las columnas
        self.tree = make_treeview(wrap, cols, height=20 if cp else 22, xscroll=cp)

    def on_show(self):
        self.refresh()

    def refresh(self):
        self._datos = _listar(self.rol_tipo)
        self._filtrar()

    def _abrir_busqueda_tactil(self, _event=None):
        """Abre una búsqueda grande con teclado virtual y filtra al aceptar."""
        _abrir_teclado(
            self.dash.root,
            "Buscar usuarios",
            self._sq,
            accept_text="Buscar",
            on_accept=lambda _value: self._filtrar(),
        )
        return "break"

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
            _touch_notice(self.dash.root, t("seleccion"), t("selecciona_reg"),
                          kind="warning")
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
            _touch_notice(self.dash.root, t("error"), t("no_autoeliminar"),
                          kind="error")
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

        _register_touch_window(self)
        self.title(titulo)
        self.configure(bg=CARD)
        self.resizable(False, False)
        self.transient(parent)
        self.bind("<Return>", lambda e: (self._save(), "break")[1])
        self.bind("<KP_Enter>", lambda e: (self._save(), "break")[1])
        self.bind("<Escape>", lambda e: (self.destroy(), "break")[1])
        self._build(titulo)
        if self._edit:
            self._prefill()
        self._finalize(parent)

    def _finalize(self, parent):
        self.update_idletasks()
        if _touch_mode_from_dashboard(self.dashboard):
            try:
                self.overrideredirect(True)
            except Exception:
                pass
        sw = max(500, parent.winfo_screenwidth())
        sh = max(500, parent.winfo_screenheight())
        if sw <= 700:
            W_DLG = min(560, sw - 12)
        else:
            W_DLG = min(560, sw - 24)
        # En Raspbian/Raspberry Pi OS no dependemos del gestor de ventanas:
        # si la pantalla es baja, el diálogo queda dentro del área visible.
        margen_h = 22 if sh <= 650 else 24
        H_DLG = min(max(self.winfo_reqheight(), 360), sh - margen_h)
        _center_on_parent(self, parent, int(W_DLG), int(H_DLG))
        self.grab_set()
        self.lift()
        try:
            self.attributes("-topmost", True)
            self.after(180, lambda: self.winfo_exists() and self.attributes("-topmost", False))
        except Exception:
            pass
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
        self._v_grado   = tk.StringVar()
        self._v_grupo   = tk.StringVar()
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
        elif self.rol_tipo == "estudiante":
            # En edición de alumnos también se permiten grado y grupo.
            row_gg = tk.Frame(form, bg=CARD)
            row_gg.pack(fill="x", pady=3)
            g1 = tk.Frame(row_gg, bg=CARD)
            g1.pack(side="left", fill="x", expand=True, padx=(0, 6))
            g2 = tk.Frame(row_gg, bg=CARD)
            g2.pack(side="left", fill="x", expand=True)
            self._lf(g1, t("grado_campo"), self._v_grado)
            self._lf(g2, t("grupo_campo"), self._v_grupo)

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
        _attach_virtual_keyboard(
            e, self, label, var, show=show,
            # Sin botón de guardar dentro del teclado.
            # Enter físico en el campo sí guarda; en táctil se guarda con el botón del formulario.
            on_submit=self._save,
        )
        return e

    def _prefill(self):
        d = self.datos_ex
        self._v_nombre.set(d.get("nombre", ""))
        self._v_ap.set(d.get("ap", ""))
        self._v_am.set(d.get("am", ""))
        self._v_cuenta.set(d.get("cuenta", ""))
        self._v_correo.set(d.get("correo", ""))
        self._v_grado.set(str(d.get("grado") or ""))
        self._v_grupo.set(str(d.get("grupo") or ""))

    def _save(self):
        self._v_err.set("")
        nombre = self._v_nombre.get().strip()
        ap     = self._v_ap.get().strip()
        am     = self._v_am.get().strip()
        cuenta = self._v_cuenta.get().strip()
        correo = self._v_correo.get().strip()
        pwd    = self._v_pwd.get()
        grado  = self._v_grado.get().strip()
        grupo  = self._v_grupo.get().strip().upper()

        if not nombre or not ap:
            self._v_err.set(t("err_nombre")); return
        if not cuenta or not cuenta.isdigit() or len(cuenta) != 8:
            self._v_err.set(t("err_cuenta")); return
        if self.rol_tipo in ("maestro", "admin") and not correo:
            self._v_err.set(t("err_correo")); return
        if self.rol_tipo in ("maestro", "admin") and not self._edit and not pwd:
            self._v_err.set(t("err_pwd")); return
        if self.rol_tipo == "estudiante":
            if not grado or not grado.isdigit():
                self._v_err.set(t("err_grado")); return
            if not grupo or len(grupo) != 1 or not grupo.isalpha():
                self._v_err.set(t("err_grupo")); return

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
        if self.rol_tipo == "estudiante":
            datos["grado"] = int(grado)
            datos["grupo"] = grupo
        editor = self.dashboard.usuario.get("id", 0)
        ok, msg = (_editar(datos, editor) if self._edit
                   else _crear(datos, editor))

        if ok:
            if self._edit:
                if self.on_save:
                    self.on_save()
                self.destroy()
                _touch_notice(self.dashboard.root, t("exito"), msg,
                              kind="success")
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
            _touch_notice(self.dashboard.root, t("error"),
                          "No se encontró el registro recién creado.",
                          kind="error")
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

        _register_touch_window(self)
        self.title(t("agregar_alumno"))
        self.configure(bg=CARD)
        self.resizable(False, False)
        self.transient(parent)
        self.bind("<Return>", lambda e: (self._continuar(), "break")[1])
        self.bind("<KP_Enter>", lambda e: (self._continuar(), "break")[1])
        self.bind("<Escape>", lambda e: (self.destroy(), "break")[1])

        self._build()
        self._finalize(parent)

    def _finalize(self, parent):
        self.update_idletasks()
        if _touch_mode_from_dashboard(self.dashboard):
            try:
                self.overrideredirect(True)
            except Exception:
                pass
        sw = max(500, parent.winfo_screenwidth())
        sh = max(500, parent.winfo_screenheight())
        if sw <= 700:
            W_DLG = min(560, sw - 12)
        else:
            W_DLG = min(560, sw - 24)
        # En horizontal 1024x600 dejamos más margen para que no choque con la barra superior/inferior.
        margen_h = 22 if sh <= 650 else 24
        H_DLG = min(max(self.winfo_reqheight(), 420), sh - margen_h)
        _center_on_parent(self, parent, int(W_DLG), int(H_DLG))
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
        _attach_virtual_keyboard(e, self, label, var,
                                 # Sin botón de guardar dentro del teclado.
                                 # Enter físico en el campo sí continúa; en táctil se usa el botón del formulario.
                                 on_submit=self._continuar)
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
            _touch_notice(self.dashboard.root, t("error"),
                          "No se encontró el registro recién creado.",
                          kind="error")
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
        _register_touch_window(self)
        self.parent     = parent
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

        try:
            self.overrideredirect(True)
        except Exception:
            pass
        W_DLG, H_DLG = 380, 220
        _center_on_parent(self, parent, W_DLG, H_DLG)
        self.grab_set()
        self.lift()
        self.focus_force()

    def _confirm(self):
        ok, msg = _eliminar(self.uid)
        if ok:
            if self.on_confirm:
                self.on_confirm()
            self.destroy()
            _touch_notice(self.parent, t("exito"), msg, kind="success")
        else:
            _touch_notice(self, t("error"), msg, kind="error")