"""
lang_dict.py
============
Diccionario de traducción para LabControl Dashboard + Interfaz táctil.
Idiomas disponibles: Español (es), Inglés (en).

Uso:
    from lang_dict import t, set_lang, current_lang, LANGS

    set_lang("en")          # cambiar idioma
    label = t("resumen")    # → "Summary"
"""

_current_lang: str = "es"

LANGS: dict[str, str] = {
    "es": "Español",
    "en": "English",
}

TRANSLATIONS: dict[str, dict[str, str]] = {

    "es": {
        # ── Sidebar ───────────────────────────────────────────
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

        # ── Header ────────────────────────────────────────────
        "btn_traductor":     "🌐 EN",
        "confirmar_salida":  "¿Deseas cerrar la sesión actual?",
        "logout_titulo":     "Cerrar sesión",

        # ── Títulos / subtítulos de vistas ────────────────────
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

        # ── Resumen: KPIs ─────────────────────────────────────
        "kpi_inside":        "Personas adentro",
        "kpi_hoy":           "Accesos hoy",
        "kpi_semana":        "Accesos esta semana",

        # ── Resumen: cards ────────────────────────────────────
        "ultimos_accesos":   "Últimos 4 accesos",
        "accesos_rt":        "Accesos recientes en tiempo real",
        "accesos_recientes": "Accesos recientes",
        "movimientos_hoy":   "Movimientos del día de hoy",
        "ver_todo":          "Ver todo",
        "nadie_lab":         "Sin accesos registrados aún",
        "en_vivo":           "● EN VIVO",

        # ── Tabla resumen / accesos ───────────────────────────
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

        # ── Vista Accesos ─────────────────────────────────────
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

        # ── Estadísticas ──────────────────────────────────────
        "top_visitantes":    "Top 7 — Más accesos registrados",
        "top_sub":           "Usuarios con mayor número de ingresos",
        "col_visitas":       "visitas",
        "accesos_por_rol":   "Accesos por rol",
        "dist_visitas":      "Distribución de visitas",
        "accesos_por_hora":  "Accesos por hora del día",
        "franja_horaria":    "Franja horaria con mayor afluencia",
        "sin_datos":         "Sin datos",
        "sin_datos_semana":  "Sin datos de esta semana",

        # ── Perfil ────────────────────────────────────────────
        "num_cuenta":        "Número de cuenta",
        "correo":            "Correo",
        "activo":            "Activo",
        "actividad_mes":     "Actividad del mes",
        "accesos_mes":       "Accesos registrados este mes",
        "usuarios_mes":      "Usuarios registrados este mes",

        # ── Configuración ─────────────────────────────────────
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

        # ── Gestión (tabla) ───────────────────────────────────
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

        # ── Gestión: formulario ───────────────────────────────
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

        # ── Eliminar ──────────────────────────────────────────
        "eliminar_titulo":   "Confirmar eliminación",
        "eliminar_pregunta": "¿Seguro que deseas eliminar a\n{nombre}?\n\nEsta acción no se puede deshacer.",
        "no_autoeliminar":   "No puedes eliminarte a ti mismo.",
        "selecciona_reg":    "Selecciona un registro de la tabla.",
        "seleccion":         "Selección",

        # ── Horario: errores ──────────────────────────────────
        "err_hora":          "Formato inválido. Usa HH:MM (p. ej. 08:00).",

        # ── Nombres de días y meses ───────────────────────────
        "day_0": "Lun", "day_1": "Mar", "day_2": "Mié",
        "day_3": "Jue", "day_4": "Vie", "day_5": "Sáb", "day_6": "Dom",
        "mon_1":  "Ene",  "mon_2":  "Feb",  "mon_3":  "Mar",
        "mon_4":  "Abr",  "mon_5":  "May",  "mon_6":  "Jun",
        "mon_7":  "Jul",  "mon_8":  "Ago",  "mon_9":  "Sep",
        "mon_10": "Oct",  "mon_11": "Nov",  "mon_12": "Dic",

        # ═══════ NUEVAS CLAVES PARA interfaz.py ══════════════
        # Menú principal
        "main_title":                   "SICEUC",
        "main_system":                  "SISTEMA DE CONTROL DE ACCESO FACIAL",
        "main_facultad":                "Facultad de Ingeniería Electromecánica",
        "main_subtitle":                "Selecciona una opción",
        "main_desc":                    "Sistema de identificación biométrica — Universidad de Colima",
        "main_btn_register":            "Registrar",
        "main_btn_register_sub":        "Nuevo usuario",
        "main_btn_register_desc":       "Captura biométrica guiada en 4 ángulos",
        "main_btn_access":              "Acceso",
        "main_btn_access_sub":          "Verificar identidad",
        "main_btn_access_desc":         "Reconocimiento facial en tiempo real",
        "main_btn_dashboard":           "Dashboard",
        "main_btn_dashboard_sub":       "Panel de administración",
        "main_btn_dashboard_desc":      "Gestión de usuarios, registros y estadísticas",
        "main_btn_enter":               "ENTRAR  ▶",
        "main_btn_close":               "✕  Salir",
        "main_btn_menu":                "☰  Menú",

        # Registro
        "reg_header":                   "REGISTRO",
        "reg_initial_pill":             "Posiciónate frente a la cámara",
        "reg_scan_btn_ready":           "⬤  INICIAR ESCANEO",
        "reg_scan_btn_locked":          "🔒  Completa el formulario para continuar",
        "reg_form_title":               "📋  Datos del usuario",
        "reg_section_identity":         "Identidad",
        "reg_field_nombre":             "Nombre(s)",
        "reg_field_ap_pat":             "Ap. paterno",
        "reg_field_ap_mat":             "Ap. materno",
        "reg_field_cuenta":             "Número de cuenta (8 dígitos)",
        "reg_section_rol":              "Rol",
        "reg_status_incomplete":        "Estado: Formulario incompleto",
        "reg_status_name":              "Escribe nombre y apellido paterno",
        "reg_status_cuenta":            "Cuenta: {msg}",
        "reg_status_correo":            "Correo: {msg}",
        "reg_status_pwd":               "Contraseña: {msg}",
        "reg_status_grado":             "Grado: {msg}",
        "reg_status_grupo":             "Grupo: {msg}",
        "reg_status_complete":          "✓  Formulario completo — puedes iniciar el escaneo",
        "reg_cancel_btn":               "✕  Cancelar registro",
        "reg_fab_tooltip":              "Datos del usuario",

        # Registro: pasos (nombres cortos)
        "step_front":                   "Frente",
        "step_left":                    "Izq.",
        "step_right":                   "Der.",

        # Registro: mensajes de progreso
        "scan_step_instruction":        "{paso}/{total}: {instruccion}",
        "scan_occluded":                "OBSTRUIDO — {razon}",
        "scan_waiting":                 "ESPERANDO — {correccion}",
        "scan_detected":                "Detectado",
        "scan_searching":               "Buscando...",
        "scan_analyzing":               "Analizando...",
        "scan_denied":                  "Desconocido",
        "scan_duplicate":               "YA REGISTRADO\n{nombre}",
        "scan_success":                 "Listo. {nombre}\n{muestras} muestras.",
        "scan_fail":                    "Inténtalo de nuevo.",
        "scan_pasos_incompletos":       "Pasos incompletos:\n{lista}",

        # Auth dialog
        "auth_title":                   "Acceso Restringido",
        "auth_subtitle":                "Solo administradores y maestros",
        "auth_field_cuenta":            "Número de cuenta",
        "auth_field_pwd":               "Contraseña",
        "auth_error_empty":             "Completa ambos campos.",
        "auth_error_notfound":          "Usuario no encontrado o sin permisos.",
        "auth_error_wrongpwd":          "Contraseña incorrecta.",
        "auth_error_conn":              "Error de conexión: {error}",
        "auth_btn_cancel":              "Cancelar",
        "auth_btn_ingresar":            "Ingresar  ▶",

        # Acceso
        "acc_header":                   "ACCESO",
        "acc_esperando":                "Esperando...",
        "acc_similitud":                "SIMILITUD",
        "acc_pill_esperando":           "Esperando...",
        "acc_pill_listo":               "✓  Listo · mira a la cámara",
        "acc_pill_acercate":            "Acércate a la cámara",
        "acc_pill_volteado":            "Estás volteado — mira al frente",
        "acc_scan_escaneando":          "Escaneando...",
        "acc_no_rostro":                "Sin rostro",
        "acc_gira_frente":              "Gira al frente",
        "acc_mira_directo":             "Mira directo a la cámara.",
        "acc_ponte_frente":             "Ponte frente a la cámara.",
        "acc_denegado":                 "ACCESO DENEGADO",
        "acc_sin_registros":            "Sin registros",
        "acc_persona_no_reconocida":    "Persona no reconocida.",
        "acc_no_hay_usuarios":          "No hay usuarios registrados.",
        "acc_estado_denegado":          "DENEGADO ✕",
        "acc_pill_denegado":            "Acceso denegado",
        "acc_angulo_match":             "Ángulo match: {angulo}",
        "acc_permitido":                "ACCESO PERMITIDO",
        "acc_estado_permitido":         "PERMITIDO ✓",
        "acc_laboratorio_lleno":        "LABORATORIO LLENO ({dentro}/{capacidad})",
        "acc_estado_lleno":             "LLENO {dentro}/{capacidad}",
        "acc_detalle_persona":          "Cuenta: {cuenta} · {rol}",
        "acc_esperando_persona":        "Esperando persona...",
        "acc_acercate_identificar":     "Acércate para identificarte",
    },

    "en": {
        # ── Sidebar ───────────────────────────────────────────
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

        # ── Header ────────────────────────────────────────────
        "btn_traductor":     "🌐 ES",
        "confirmar_salida":  "Do you want to end the current session?",
        "logout_titulo":     "Log out",

        # ── View titles ───────────────────────────────────────
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

        # ── KPIs ──────────────────────────────────────────────
        "kpi_inside":        "People inside",
        "kpi_hoy":           "Accesses today",
        "kpi_semana":        "Accesses this week",

        # ── Summary cards ─────────────────────────────────────
        "ultimos_accesos":   "Last 4 accesses",
        "accesos_rt":        "Recent accesses in real time",
        "accesos_recientes": "Recent accesses",
        "movimientos_hoy":   "Today's movements",
        "ver_todo":          "View all",
        "nadie_lab":         "No accesses recorded yet",
        "en_vivo":           "● LIVE",

        # ── Table columns ─────────────────────────────────────
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

        # ── Access log view ───────────────────────────────────
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

        # ── Statistics ────────────────────────────────────────
        "top_visitantes":    "Top 7 — Most accesses",
        "top_sub":           "Users with the highest number of entries",
        "col_visitas":       "visits",
        "accesos_por_rol":   "Accesses by role",
        "dist_visitas":      "Visit distribution",
        "accesos_por_hora":  "Accesses by time of day",
        "franja_horaria":    "Peak access hour",
        "sin_datos":         "No data",
        "sin_datos_semana":  "No data for this week",

        # ── Profile ───────────────────────────────────────────
        "num_cuenta":        "Account number",
        "correo":            "Email",
        "activo":            "Active",
        "actividad_mes":     "Monthly activity",
        "accesos_mes":       "Accesses logged this month",
        "usuarios_mes":      "Users registered this month",

        # ── Settings ──────────────────────────────────────────
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

        # ── Management table ──────────────────────────────────
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

        # ── Form ──────────────────────────────────────────────
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

        # ── Delete ────────────────────────────────────────────
        "eliminar_titulo":   "Confirm deletion",
        "eliminar_pregunta": "Are you sure you want to delete\n{nombre}?\n\nThis action cannot be undone.",
        "no_autoeliminar":   "You cannot delete your own account.",
        "selecciona_reg":    "Select a record from the table.",
        "seleccion":         "Selection",

        # ── Schedule errors ───────────────────────────────────
        "err_hora":          "Invalid format. Use HH:MM (e.g. 08:00).",

        # ── Date names ────────────────────────────────────────
        "day_0": "Mon", "day_1": "Tue", "day_2": "Wed",
        "day_3": "Thu", "day_4": "Fri", "day_5": "Sat", "day_6": "Sun",
        "mon_1":  "Jan",  "mon_2":  "Feb",  "mon_3":  "Mar",
        "mon_4":  "Apr",  "mon_5":  "May",  "mon_6":  "Jun",
        "mon_7":  "Jul",  "mon_8":  "Aug",  "mon_9":  "Sep",
        "mon_10": "Oct",  "mon_11": "Nov",  "mon_12": "Dec",

        # ═══════ NEW KEYS FOR interfaz.py ════════════════════
        # Main menu
        "main_title":                   "SICEUC",
        "main_system":                  "FACIAL ACCESS CONTROL SYSTEM",
        "main_facultad":                "Faculty of Electromechanical Engineering",
        "main_subtitle":                "Select an option",
        "main_desc":                    "Biometric identification system — University of Colima",
        "main_btn_register":            "Register",
        "main_btn_register_sub":        "New user",
        "main_btn_register_desc":       "Guided biometric capture in 4 angles",
        "main_btn_access":              "Access",
        "main_btn_access_sub":          "Verify identity",
        "main_btn_access_desc":         "Real-time facial recognition",
        "main_btn_dashboard":           "Dashboard",
        "main_btn_dashboard_sub":       "Administration panel",
        "main_btn_dashboard_desc":      "User, log, and statistics management",
        "main_btn_enter":               "ENTER  ▶",
        "main_btn_close":               "✕  Exit",
        "main_btn_menu":                "☰  Menu",

        # Registration
        "reg_header":                   "REGISTRATION",
        "reg_initial_pill":             "Position yourself in front of the camera",
        "reg_scan_btn_ready":           "⬤  START SCAN",
        "reg_scan_btn_locked":          "🔒  Complete the form to continue",
        "reg_form_title":               "📋  User data",
        "reg_section_identity":         "Identity",
        "reg_field_nombre":             "First name(s)",
        "reg_field_ap_pat":             "First surname",
        "reg_field_ap_mat":             "Second surname",
        "reg_field_cuenta":             "Account number (8 digits)",
        "reg_section_rol":              "Role",
        "reg_status_incomplete":        "Status: Incomplete form",
        "reg_status_name":              "Enter first name and first surname",
        "reg_status_cuenta":            "Account: {msg}",
        "reg_status_correo":            "Email: {msg}",
        "reg_status_pwd":               "Password: {msg}",
        "reg_status_grado":             "Grade: {msg}",
        "reg_status_grupo":             "Group: {msg}",
        "reg_status_complete":          "✓  Form complete — you can start the scan",
        "reg_cancel_btn":               "✕  Cancel registration",
        "reg_fab_tooltip":              "User data",

        # Registration: step labels
        "step_front":                   "Front",
        "step_left":                    "Left",
        "step_right":                   "Right",

        # Registration: scan messages
        "scan_step_instruction":        "{paso}/{total}: {instruccion}",
        "scan_occluded":                "OBSTRUCTED — {razon}",
        "scan_waiting":                 "WAITING — {correccion}",
        "scan_detected":                "Detected",
        "scan_searching":               "Searching...",
        "scan_analyzing":               "Analyzing...",
        "scan_denied":                  "Unknown",
        "scan_duplicate":               "ALREADY REGISTERED\n{nombre}",
        "scan_success":                 "Done. {nombre}\n{muestras} samples.",
        "scan_fail":                    "Try again.",
        "scan_pasos_incompletos":       "Incomplete steps:\n{lista}",

        # Auth dialog
        "auth_title":                   "Restricted Access",
        "auth_subtitle":                "Administrators and teachers only",
        "auth_field_cuenta":            "Account number",
        "auth_field_pwd":               "Password",
        "auth_error_empty":             "Fill in both fields.",
        "auth_error_notfound":          "User not found or no permissions.",
        "auth_error_wrongpwd":          "Incorrect password.",
        "auth_error_conn":              "Connection error: {error}",
        "auth_btn_cancel":              "Cancel",
        "auth_btn_ingresar":            "Log in  ▶",

        # Access
        "acc_header":                   "ACCESS",
        "acc_esperando":                "Waiting...",
        "acc_similitud":                "SIMILARITY",
        "acc_pill_esperando":           "Waiting...",
        "acc_pill_listo":               "✓  Ready · look at the camera",
        "acc_pill_acercate":            "Move closer to the camera",
        "acc_pill_volteado":            "You are turned away — face the camera",
        "acc_scan_escaneando":          "Scanning...",
        "acc_no_rostro":                "No face",
        "acc_gira_frente":              "Turn to the front",
        "acc_mira_directo":             "Look straight at the camera.",
        "acc_ponte_frente":             "Stand in front of the camera.",
        "acc_denegado":                 "ACCESS DENIED",
        "acc_sin_registros":            "No records",
        "acc_persona_no_reconocida":    "Person not recognized.",
        "acc_no_hay_usuarios":          "No registered users.",
        "acc_estado_denegado":          "DENIED ✕",
        "acc_pill_denegado":            "Access denied",
        "acc_angulo_match":             "Match angle: {angulo}",
        "acc_permitido":                "ACCESS GRANTED",
        "acc_estado_permitido":         "GRANTED ✓",
        "acc_laboratorio_lleno":        "LAB FULL ({dentro}/{capacidad})",
        "acc_estado_lleno":             "FULL {dentro}/{capacidad}",
        "acc_detalle_persona":          "Account: {cuenta} · {rol}",
        "acc_esperando_persona":        "Waiting for a person...",
        "acc_acercate_identificar":     "Approach to identify yourself",
    },
}


# ══════════════════════════════════════════════════════════════
#  API pública
# ══════════════════════════════════════════════════════════════

def set_lang(lang: str) -> None:
    global _current_lang
    if lang not in TRANSLATIONS:
        raise ValueError(f"Idioma '{lang}' no disponible. Opciones: {list(TRANSLATIONS)}")
    _current_lang = lang


def current_lang() -> str:
    return _current_lang


def toggle_lang() -> str:
    langs = list(TRANSLATIONS.keys())
    idx   = langs.index(_current_lang)
    nxt   = langs[(idx + 1) % len(langs)]
    set_lang(nxt)
    return nxt


def t(key: str, **kwargs) -> str:
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
    from datetime import datetime
    d = dt or datetime.now()
    dia_semana = t(f"day_{d.weekday()}")
    mes        = t(f"mon_{d.month}")
    return f"{dia_semana} {d.day:02d} {mes} {d.year}"