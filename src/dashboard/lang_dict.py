"""
lang_dict.py
============
Diccionario de traducción para LabControl Dashboard.
Idiomas disponibles: Español (es), Inglés (en).

Uso:
    from lang_dict import t, set_lang, current_lang, LANGS

    set_lang("en")          # cambiar idioma
    label = t("resumen")    # → "Summary"

Para agregar un nuevo idioma:
    1. Añade una entrada en TRANSLATIONS con el código ISO.
    2. Copia todas las claves del diccionario "es" y tradúcelas.
"""

# Idioma activo (estado global de la sesión)
_current_lang: str = "es"

LANGS: dict[str, str] = {
    "es": "Español",
    "en": "English",
}

TRANSLATIONS: dict[str, dict[str, str]] = {

    # ══════════════════════════════════════════════════════════════
    # ESPAÑOL  (fuente de verdad — todas las claves deben existir aquí)
    # ══════════════════════════════════════════════════════════════
    "es": {

        # ── Sidebar ───────────────────────────────────────────────
        "sistema_gestion":   "Sistema de Gestión",
        "nav_panel":         "Panel",
        "nav_resumen":       "📊  Resumen",
        "nav_inside":        "👥  En el laboratorio",
        "nav_accesos":       "📋  Registro de acceso",
        "nav_stats":         "📈  Estadísticas",
        "nav_gestion":       "Gestión",
        "nav_alumnos":       "🎓  Alumnos",
        "nav_maestros":      "👨‍🏫  Maestros",
        "nav_admins":        "🛡️   Administradores",
        "nav_cuenta":        "Cuenta",
        "nav_perfil":        "👤  Mi perfil",
        "nav_config":        "⚙️   Configuración",
        "cerrar_sesion":     "⏻  Cerrar sesión",
        "administrador":     "Administrador",
        "maestro":           "Maestro",
        "alumno":            "Alumno",

        # ── Header ────────────────────────────────────────────────
        "btn_traductor":     "🌐 EN",   # texto del botón cuando idioma activo es ES
        "confirmar_salida":  "¿Deseas cerrar la sesión actual?",
        "logout_titulo":     "Cerrar sesión",

        # ── Títulos / subtítulos de vistas ────────────────────────
        "title_resumen":     "Resumen del laboratorio",
        "sub_resumen":       "Vista general en tiempo real",
        "title_inside":      "En el laboratorio",
        "sub_inside":        "Presencia actual en tiempo real",
        "title_accesos":     "Registro de accesos",
        "sub_accesos":       "Historial de entradas al laboratorio",
        "title_stats":       "Estadísticas de uso",
        "sub_stats":         "Análisis de visitas y tendencias",
        "title_alumnos":     "Gestión de Alumnos",
        "sub_alumnos":       "Registra, edita y elimina alumnos",
        "title_maestros":    "Gestión de Maestros",
        "sub_maestros":      "Administra los maestros del sistema",
        "title_admins":      "Gestión de Administradores",
        "sub_admins":        "Administra los administradores",
        "title_perfil":      "Mi perfil",
        "sub_perfil":        "Información de tu cuenta",
        "title_config":      "Configuración",
        "sub_config":        "Preferencias del sistema",

        # ── Resumen: KPIs ─────────────────────────────────────────
        "kpi_inside":        "Personas adentro",
        "kpi_hoy":           "Accesos hoy",
        "kpi_semana":        "Accesos esta semana",

        # ── Resumen: cards ────────────────────────────────────────
        "ultimos_accesos":   "Últimos 4 accesos",
        "accesos_rt":        "Accesos recientes en tiempo real",
        "accesos_recientes": "Accesos recientes",
        "movimientos_hoy":   "Movimientos del día de hoy",
        "ver_todo":          "Ver todo",
        "nadie_lab":         "Sin accesos registrados aún",
        "en_vivo":           "● EN VIVO",

        # ── Tabla resumen / accesos ───────────────────────────────
        "col_persona":       "Persona",
        "col_tipo":          "Tipo",
        "col_hora":          "Hora",
        "col_rol":           "Rol",
        "col_num":           "#",
        "col_nombre":        "Nombre",
        "col_cuenta":        "No. Cuenta",
        "col_grado":         "Grado",
        "col_grupo":         "Grupo",
        "col_fecha":         "Fecha",
        "col_estado":        "Estado",
        "col_matricula":     "Matrícula",
        "entrada":           "↓ Entrada",
        "salida":            "↑ Salida",
        "dentro":            "✓ Dentro",
        "tiempo_exc":        "⚠ Tiempo excedido",

        # ── Vista Accesos ─────────────────────────────────────────
        "filtro_todos":      "Todos",
        "filtro_estudiante": "Estudiantes",
        "filtro_maestro":    "Maestros",
        "filtro_admin":      "Admins",
        "filtro_rol":        "Filtrar por rol:",
        "buscar":            "Buscar",
        "registros_encontrados": "{n} registros encontrados",
        "sin_registros":     "No hay registros con este filtro.",
        "registro_completo": "Registro completo de accesos",
        "hist_entradas":     "Historial de entradas al laboratorio",

        # ── Estadísticas ──────────────────────────────────────────
        "top_visitantes":    "Top 7 — Más accesos registrados",
        "top_sub":           "Usuarios con mayor número de ingresos",
        "col_visitas":       "visitas",
        "accesos_por_rol":   "Accesos por rol",
        "dist_visitas":      "Distribución de visitas",
        "accesos_por_hora":  "Accesos por hora del día",
        "franja_horaria":    "Franja horaria con mayor afluencia",
        "sin_datos":         "Sin datos",
        "sin_datos_semana":  "Sin datos de esta semana",

        # ── Perfil ────────────────────────────────────────────────
        "num_cuenta":        "Número de cuenta",
        "correo":            "Correo",
        "activo":            "Activo",
        "actividad_mes":     "Actividad del mes",
        "accesos_mes":       "Accesos registrados este mes",
        "usuarios_mes":      "Usuarios registrados este mes",

        # ── Configuración ─────────────────────────────────────────
        "config_sistema":    "Configuración del sistema",
        "prefs":             "Preferencias de la sesión",
        "horario_acceso":    "Horario de acceso",
        "horario_desc":      "Define el rango horario en que se permite el acceso al laboratorio",
        "btn_config_horario":"⏱  Configurar horario",
        "horario_titulo":    "Configurar horario de acceso",
        "hora_inicio":       "Hora de inicio",
        "hora_fin":          "Hora de fin (00:00 = sin límite)",
        "aplicar":           "Aplicar",
        "cancelar":          "Cancelar",
        "horario_guardado":  "Horario actualizado correctamente.",
        "horario_actual":    "Horario actual: {ini} — {fin}",
        "sin_limite":        "Sin límite",
        "info_sesion":       "Información de sesión",
        "info_sesion_desc":  "Datos del usuario actualmente logueado",
        "autoactualizacion": "Actualización automática",
        "autoactualizacion_desc": "Los datos se refrescan cada 30 segundos",
        "activo_30s":        "✓  Activo — cada 30 segundos",

        # ── Gestión (tabla) ───────────────────────────────────────
        "agregar":           "＋  Agregar",
        "editar":            "✏  Editar",
        "eliminar":          "🗑  Eliminar",
        "buscar_lbl":        "Buscar:",
        "registros":         "registro",
        "registros_pl":      "registros",
        "cargando":          "Cargando…",
        "sin_alumnos":       "No hay alumnos registrados.",
        "sin_maestros":      "No hay maestros registrados.",
        "sin_admins":        "No hay administradores registrados.",
        "col_grado_grupo":   "Grado / Grupo",
        "col_registrado":    "Registrado",
        "col_act_por":       "Actualizado por",
        "col_ult_act":       "Últ. actualiz.",

        # ── Gestión: formulario ───────────────────────────────────
        "agregar_alumno":    "Agregar Alumno",
        "editar_alumno":     "Editar Alumno",
        "agregar_maestro":   "Agregar Maestro",
        "editar_maestro":    "Editar Maestro",
        "agregar_admin":     "Agregar Administrador",
        "editar_admin":      "Editar Administrador",
        "nombres":           "Nombre(s) *",
        "ap_pat":            "Apellido paterno *",
        "ap_mat":            "Apellido materno",
        "cuenta_campo":      "Número de cuenta (8 dígitos) *",
        "correo_campo":      "Correo institucional *",
        "grado_campo":       "Grado *",
        "grupo_campo":       "Grupo (A-Z) *",
        "pwd_nuevo":         "Contraseña *",
        "pwd_editar":        "Nueva contraseña (dejar vacío = sin cambio)",
        "guardar":           "Guardar",
        "err_nombre":        "⚠ Nombre y apellido paterno son obligatorios.",
        "err_cuenta":        "⚠ Número de cuenta: 8 dígitos numéricos.",
        "err_correo":        "⚠ Correo obligatorio para maestros y admins.",
        "err_pwd":           "⚠ Contraseña obligatoria al crear un nuevo usuario.",
        "err_grado":         "⚠ Grado debe ser un número.",
        "err_grupo":         "⚠ Grupo debe ser una letra (A-Z).",
        "exito":             "Éxito",
        "error":             "Error",

        # ── Eliminar ──────────────────────────────────────────────
        "eliminar_titulo":   "Confirmar eliminación",
        "eliminar_pregunta": "¿Seguro que deseas eliminar a\n{nombre}?\n\nEsta acción no se puede deshacer.",
        "no_autoeliminar":   "No puedes eliminarte a ti mismo.",
        "selecciona_reg":    "Selecciona un registro de la tabla.",
        "seleccion":         "Selección",

        # ── Horario: errores ──────────────────────────────────────
        "err_hora":          "Formato inválido. Usa HH:MM (p. ej. 08:00).",

        # ── Nombres de días y meses ───────────────────────────────
        "day_0": "Lun", "day_1": "Mar", "day_2": "Mié",
        "day_3": "Jue", "day_4": "Vie", "day_5": "Sáb", "day_6": "Dom",
        "mon_1":  "Ene",  "mon_2":  "Feb",  "mon_3":  "Mar",
        "mon_4":  "Abr",  "mon_5":  "May",  "mon_6":  "Jun",
        "mon_7":  "Jul",  "mon_8":  "Ago",  "mon_9":  "Sep",
        "mon_10": "Oct",  "mon_11": "Nov",  "mon_12": "Dic",
    },

    # ══════════════════════════════════════════════════════════════
    # ENGLISH
    # ══════════════════════════════════════════════════════════════
    "en": {

        # ── Sidebar ───────────────────────────────────────────────
        "sistema_gestion":   "Management System",
        "nav_panel":         "Panel",
        "nav_resumen":       "📊  Summary",
        "nav_inside":        "👥  In the lab",
        "nav_accesos":       "📋  Access log",
        "nav_stats":         "📈  Statistics",
        "nav_gestion":       "Management",
        "nav_alumnos":       "🎓  Students",
        "nav_maestros":      "👨‍🏫  Teachers",
        "nav_admins":        "🛡️   Admins",
        "nav_cuenta":        "Account",
        "nav_perfil":        "👤  My profile",
        "nav_config":        "⚙️   Settings",
        "cerrar_sesion":     "⏻  Log out",
        "administrador":     "Administrator",
        "maestro":           "Teacher",
        "alumno":            "Student",

        # ── Header ────────────────────────────────────────────────
        "btn_traductor":     "🌐 ES",
        "confirmar_salida":  "Do you want to end the current session?",
        "logout_titulo":     "Log out",

        # ── View titles ───────────────────────────────────────────
        "title_resumen":     "Lab summary",
        "sub_resumen":       "Real-time overview",
        "title_inside":      "In the laboratory",
        "sub_inside":        "Current presence in real time",
        "title_accesos":     "Access log",
        "sub_accesos":       "Entry history",
        "title_stats":       "Usage statistics",
        "sub_stats":         "Visit analysis and trends",
        "title_alumnos":     "Student Management",
        "sub_alumnos":       "Register, edit and delete students",
        "title_maestros":    "Teacher Management",
        "sub_maestros":      "Manage system teachers",
        "title_admins":      "Admin Management",
        "sub_admins":        "Manage system administrators",
        "title_perfil":      "My profile",
        "sub_perfil":        "Your account information",
        "title_config":      "Settings",
        "sub_config":        "System preferences",

        # ── KPIs ──────────────────────────────────────────────────
        "kpi_inside":        "People inside",
        "kpi_hoy":           "Accesses today",
        "kpi_semana":        "Accesses this week",

        # ── Summary cards ─────────────────────────────────────────
        "ultimos_accesos":   "Last 4 accesses",
        "accesos_rt":        "Recent accesses in real time",
        "accesos_recientes": "Recent accesses",
        "movimientos_hoy":   "Today's movements",
        "ver_todo":          "View all",
        "nadie_lab":         "No accesses recorded yet",
        "en_vivo":           "● LIVE",

        # ── Table columns ─────────────────────────────────────────
        "col_persona":       "Person",
        "col_tipo":          "Type",
        "col_hora":          "Time",
        "col_rol":           "Role",
        "col_num":           "#",
        "col_nombre":        "Name",
        "col_cuenta":        "Account No.",
        "col_grado":         "Grade",
        "col_grupo":         "Group",
        "col_fecha":         "Date",
        "col_estado":        "Status",
        "col_matricula":     "ID number",
        "entrada":           "↓ Entry",
        "salida":            "↑ Exit",
        "dentro":            "✓ Inside",
        "tiempo_exc":        "⚠ Time exceeded",

        # ── Access log view ───────────────────────────────────────
        "filtro_todos":      "All",
        "filtro_estudiante": "Students",
        "filtro_maestro":    "Teachers",
        "filtro_admin":      "Admins",
        "filtro_rol":        "Filter by role:",
        "buscar":            "Search",
        "registros_encontrados": "{n} records found",
        "sin_registros":     "No records with this filter.",
        "registro_completo": "Full access log",
        "hist_entradas":     "Entry history for the laboratory",

        # ── Statistics ────────────────────────────────────────────
        "top_visitantes":    "Top 7 — Most accesses",
        "top_sub":           "Users with the highest number of entries",
        "col_visitas":       "visits",
        "accesos_por_rol":   "Accesses by role",
        "dist_visitas":      "Visit distribution",
        "accesos_por_hora":  "Accesses by time of day",
        "franja_horaria":    "Peak access hour",
        "sin_datos":         "No data",
        "sin_datos_semana":  "No data for this week",

        # ── Profile ───────────────────────────────────────────────
        "num_cuenta":        "Account number",
        "correo":            "Email",
        "activo":            "Active",
        "actividad_mes":     "Monthly activity",
        "accesos_mes":       "Accesses logged this month",
        "usuarios_mes":      "Users registered this month",

        # ── Settings ──────────────────────────────────────────────
        "config_sistema":    "System settings",
        "prefs":             "Session preferences",
        "horario_acceso":    "Access schedule",
        "horario_desc":      "Define the time range during which lab access is allowed",
        "btn_config_horario":"⏱  Configure schedule",
        "horario_titulo":    "Configure access schedule",
        "hora_inicio":       "Start time",
        "hora_fin":          "End time (00:00 = no limit)",
        "aplicar":           "Apply",
        "cancelar":          "Cancel",
        "horario_guardado":  "Schedule updated successfully.",
        "horario_actual":    "Current schedule: {ini} — {fin}",
        "sin_limite":        "No limit",
        "info_sesion":       "Session information",
        "info_sesion_desc":  "Currently logged-in user details",
        "autoactualizacion": "Auto-refresh",
        "autoactualizacion_desc": "Data refreshes every 30 seconds",
        "activo_30s":        "✓  Active — every 30 seconds",

        # ── Management table ──────────────────────────────────────
        "agregar":           "＋  Add",
        "editar":            "✏  Edit",
        "eliminar":          "🗑  Delete",
        "buscar_lbl":        "Search:",
        "registros":         "record",
        "registros_pl":      "records",
        "cargando":          "Loading…",
        "sin_alumnos":       "No students registered.",
        "sin_maestros":      "No teachers registered.",
        "sin_admins":        "No admins registered.",
        "col_grado_grupo":   "Grade / Group",
        "col_registrado":    "Registered",
        "col_act_por":       "Updated by",
        "col_ult_act":       "Last updated",

        # ── Form ─────────────────────────────────────────────────
        "agregar_alumno":    "Add Student",
        "editar_alumno":     "Edit Student",
        "agregar_maestro":   "Add Teacher",
        "editar_maestro":    "Edit Teacher",
        "agregar_admin":     "Add Administrator",
        "editar_admin":      "Edit Administrator",
        "nombres":           "First name(s) *",
        "ap_pat":            "First surname *",
        "ap_mat":            "Second surname",
        "cuenta_campo":      "Account number (8 digits) *",
        "correo_campo":      "Institutional email *",
        "grado_campo":       "Grade *",
        "grupo_campo":       "Group (A-Z) *",
        "pwd_nuevo":         "Password *",
        "pwd_editar":        "New password (leave blank = no change)",
        "guardar":           "Save",
        "err_nombre":        "⚠ First name and first surname are required.",
        "err_cuenta":        "⚠ Account number: 8 numeric digits.",
        "err_correo":        "⚠ Email required for teachers and admins.",
        "err_pwd":           "⚠ Password required when creating a new user.",
        "err_grado":         "⚠ Grade must be a number.",
        "err_grupo":         "⚠ Group must be a single letter (A-Z).",
        "exito":             "Success",
        "error":             "Error",

        # ── Delete ────────────────────────────────────────────────
        "eliminar_titulo":   "Confirm deletion",
        "eliminar_pregunta": "Are you sure you want to delete\n{nombre}?\n\nThis action cannot be undone.",
        "no_autoeliminar":   "You cannot delete your own account.",
        "selecciona_reg":    "Select a record from the table.",
        "seleccion":         "Selection",

        # ── Schedule errors ───────────────────────────────────────
        "err_hora":          "Invalid format. Use HH:MM (e.g. 08:00).",

        # ── Date names ────────────────────────────────────────────
        "day_0": "Mon", "day_1": "Tue", "day_2": "Wed",
        "day_3": "Thu", "day_4": "Fri", "day_5": "Sat", "day_6": "Sun",
        "mon_1":  "Jan",  "mon_2":  "Feb",  "mon_3":  "Mar",
        "mon_4":  "Apr",  "mon_5":  "May",  "mon_6":  "Jun",
        "mon_7":  "Jul",  "mon_8":  "Aug",  "mon_9":  "Sep",
        "mon_10": "Oct",  "mon_11": "Nov",  "mon_12": "Dec",
    },
}


# ══════════════════════════════════════════════════════════════
#  API pública
# ══════════════════════════════════════════════════════════════

def set_lang(lang: str) -> None:
    """Cambia el idioma activo. Lanza ValueError si el código no existe."""
    global _current_lang
    if lang not in TRANSLATIONS:
        raise ValueError(f"Idioma '{lang}' no disponible. Opciones: {list(TRANSLATIONS)}")
    _current_lang = lang


def current_lang() -> str:
    """Devuelve el código del idioma activo ('es', 'en', …)."""
    return _current_lang


def toggle_lang() -> str:
    """
    Alterna entre los idiomas disponibles en orden.
    Devuelve el código del nuevo idioma activo.
    """
    langs = list(TRANSLATIONS.keys())
    idx   = langs.index(_current_lang)
    nxt   = langs[(idx + 1) % len(langs)]
    set_lang(nxt)
    return nxt


def t(key: str, **kwargs) -> str:
    """
    Devuelve la cadena traducida para *key* en el idioma activo.

    Si la clave no existe en el idioma activo, intenta con 'es'.
    Si tampoco existe, devuelve la propia clave como fallback visible.

    Soporte de interpolación via str.format():
        t("registros_encontrados", n=42)  →  "42 registros encontrados"
    """
    text = (TRANSLATIONS.get(_current_lang, {}).get(key)
            or TRANSLATIONS.get("es", {}).get(key)
            or key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass
    return text


def fecha_local(dt=None) -> str:
    """
    Devuelve la fecha en el idioma activo, p. ej.:
        ES → "Lun 14 Abr 2025"
        EN → "Mon 14 Apr 2025"
    Si dt es None usa datetime.now().
    """
    from datetime import datetime
    d = dt or datetime.now()
    dia_semana = t(f"day_{d.weekday()}")
    mes        = t(f"mon_{d.month}")
    return f"{dia_semana} {d.day:02d} {mes} {d.year}"