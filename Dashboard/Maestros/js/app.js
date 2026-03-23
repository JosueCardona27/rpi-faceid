/* ═══════════════════════════════════════════════════════
   app.js — Lógica principal del dashboard.
═══════════════════════════════════════════════════════ */

/* ================================================================
   4. RENDERIZADO DEL DASHBOARD
================================================================ */

/** Llama a todas las funciones de render en orden */
function renderAll() {
  updateKPIs();
  renderInsideListDash();
  renderInsideTable();
  renderAccessTable();
  renderRecentTable();
  renderTopVisitors();
  renderCareerChart();
  renderHourChart();
  updateSidebarBadge();
}

/* ── 4a. Tarjetas KPI ── */
function updateKPIs() {
  // Personas dentro ahora
  setEl('kpi-inside', INSIDE_NOW.length);

  // Total de entradas del día
  const entries = ACCESS_LOG.filter(r => r.type === 'Entrada').length;
  setEl('kpi-today', entries);

  // Tiempo promedio de permanencia (basado en registros con salida)
  const withDuration = ACCESS_LOG.filter(r => r.duration);
  if (withDuration.length > 0) {
    const avgMins = Math.round(
      withDuration.reduce((acc, r) => acc + parseDuration(r.duration), 0) / withDuration.length
    );
    setEl('kpi-avg', formatDuration(avgMins));
  } else {
    setEl('kpi-avg', '—');
  }
  // El KPI de alertas lo actualiza checkAlerts()
}

/* ── 4b. Lista de personas dentro — widget del dashboard ── */
function renderInsideListDash() {
  const container = document.getElementById('inside-list-dash');
  if (!container) return;

  if (INSIDE_NOW.length === 0) {
    container.innerHTML = `
      <div class="empty-state-sm" style="padding:32px 16px">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="28" height="28">
          <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
          <circle cx="9" cy="7" r="4"/>
        </svg>
        <span>No hay personas en el laboratorio</span>
      </div>`;
    return;
  }

  container.innerHTML = INSIDE_NOW.map(p => {
    const mins     = minutesElapsed(p.entryTime);
    const isAlert  = mins >= TIME_LIMIT_MINUTES;
    const pct      = Math.min(100, Math.round((mins / TIME_LIMIT_MINUTES) * 100));
    const barClass = pct >= 100 ? 'alert' : pct >= 80 ? 'warn' : '';

    return `
      <div class="inside-row">
        <div class="ir-avatar" style="${isAlert ? 'background:var(--red-dim);color:var(--red)' : ''}">${p.initials}</div>
        <div class="ir-info">
          <div class="ir-name">${p.name}</div>
          <div class="ir-career">${p.career}</div>
          <div class="time-progress" style="margin-top:5px">
            <div class="tp-track"><div class="tp-fill ${barClass}" style="width:${pct}%"></div></div>
            <div class="tp-pct">${pct}%</div>
          </div>
        </div>
        <div class="ir-time ${isAlert ? 'warn' : ''}">${formatDuration(mins)}</div>
      </div>`;
  }).join('');
}

/* ── 4c. Tabla completa — vista "En el laboratorio" ── */
function renderInsideTable() {
  const tbody = document.getElementById('inside-tbody');
  if (!tbody) return;

  if (INSIDE_NOW.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="table-empty">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="32" height="32" style="color:var(--text-3)">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
        <circle cx="9" cy="7" r="4"/>
      </svg>
      Nadie en el laboratorio en este momento
    </td></tr>`;
    return;
  }

  tbody.innerHTML = INSIDE_NOW.map(p => {
    const mins     = minutesElapsed(p.entryTime);
    const isAlert  = mins >= TIME_LIMIT_MINUTES;
    const pct      = Math.min(100, Math.round((mins / TIME_LIMIT_MINUTES) * 100));
    const barClass = pct >= 100 ? 'alert' : pct >= 80 ? 'warn' : '';

    return `
      <tr class="${isAlert ? 'row-alert' : ''}">
        <td class="td-name">
          <div style="display:flex;align-items:center;gap:9px">
            <div class="ir-avatar" style="width:30px;height:30px;font-size:10px;${isAlert ? 'background:var(--red-dim);color:var(--red)' : ''}">${p.initials}</div>
            ${p.name}
          </div>
        </td>
        <td class="td-mono">${p.matricula}</td>
        <td>${p.career}</td>
        <td class="td-mono">${formatTime(p.entryTime)}</td>
        <td>
          <div class="time-progress">
            <div class="tp-track"><div class="tp-fill ${barClass}" style="width:${pct}%"></div></div>
            <span class="td-mono" style="font-size:11px;margin-left:4px">${formatDuration(mins)}</span>
          </div>
        </td>
        <td>${isAlert
          ? '<span class="badge warn">⚠ Tiempo excedido</span>'
          : '<span class="status-dot inside">Dentro</span>'}</td>
      </tr>`;
  }).join('');
}

/* ── 4d. Tabla registro de accesos (con filtros de tipo y carrera) ── */
function renderAccessTable() {
  const tbody   = document.getElementById('access-tbody');
  const typeF   = (document.getElementById('filter-type')   || {}).value || '';
  const careerF = (document.getElementById('filter-career') || {}).value || '';
  if (!tbody) return;

  let rows = [...ACCESS_LOG].reverse();  // más reciente primero

  if (typeF)   rows = rows.filter(r => r.type === typeF);
  if (careerF) rows = rows.filter(r => r.career === careerF);

  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="table-empty">No se encontraron registros con los filtros aplicados.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map((r, i) => `
    <tr>
      <td class="td-num">${rows.length - i}</td>
      <td class="td-name">${r.name}</td>
      <td class="td-mono">${r.matricula}</td>
      <td>${r.career}</td>
      <td><span class="badge ${r.type === 'Entrada' ? 'entrada' : 'salida'}">${r.type === 'Entrada' ? '↓ Entrada' : '↑ Salida'}</span></td>
      <td class="td-mono">${r.time}</td>
      <td>${r.duration ? `<span class="dur-chip">${r.duration}</span>` : '<span style="color:var(--text-3)">—</span>'}</td>
    </tr>`).join('');
}

/* ── 4e. Tabla de accesos recientes (mini, en el dashboard) ── */
function renderRecentTable() {
  const tbody = document.getElementById('recent-tbody');
  if (!tbody) return;

  const recent = [...ACCESS_LOG].reverse().slice(0, 6);  // últimos 6 movimientos

  tbody.innerHTML = recent.map(r => `
    <tr>
      <td class="td-name">${r.name}</td>
      <td><span class="badge ${r.type === 'Entrada' ? 'entrada' : 'salida'}">${r.type === 'Entrada' ? '↓ Entrada' : '↑ Salida'}</span></td>
      <td class="td-mono">${r.time}</td>
      <td><span class="status-dot ${r.type === 'Entrada' ? 'inside' : 'out'}">${r.type === 'Entrada' ? 'Dentro' : 'Salió'}</span></td>
    </tr>`).join('');
}

/* ── 4f. Estadísticas ── */

/** Top visitantes del mes (lista con ranking) */
function renderTopVisitors() {
  const container = document.getElementById('top-list');
  if (!container) return;

  const rankClass = ['gold', 'silver', 'bronze'];

  container.innerHTML = TOP_VISITORS.map((v, i) => `
    <div class="top-item">
      <div class="top-rank ${rankClass[i] || ''}">#${i + 1}</div>
      <div class="top-av">${v.initials}</div>
      <div class="top-info">
        <div class="top-name">${v.name}</div>
        <div class="top-career">${v.career}</div>
      </div>
      <div class="top-count">${v.visits} visitas</div>
    </div>`).join('');
}

/** Gráfica de barras por carrera (CSS puro, sin librería) */
function renderCareerChart() {
  const container = document.getElementById('career-chart');
  if (!container) return;

  const max = Math.max(...CAREER_STATS.map(c => c.count));

  container.innerHTML = CAREER_STATS.map(c => `
    <div class="bar-row">
      <div class="bar-label">${c.label}</div>
      <div class="bar-track">
        <div class="bar-fill ${c.color}" style="width:${Math.round((c.count / max) * 100)}%"></div>
      </div>
      <div class="bar-val">${c.count}</div>
    </div>`).join('');
}

/** Gráfica de columnas por hora del día (la hora pico se resalta) */
function renderHourChart() {
  const container = document.getElementById('hour-chart');
  if (!container) return;

  const max  = Math.max(...HOUR_STATS.map(h => h.count));
  const peak = HOUR_STATS.find(h => h.count === max);  // hora con más accesos

  container.innerHTML = HOUR_STATS.map(h => {
    const heightPct = Math.max(4, Math.round((h.count / max) * 100));
    const isPeak    = h === peak;
    return `
      <div class="hcol ${isPeak ? 'peak' : ''}">
        <div class="hcol-bar ${isPeak ? 'peak' : ''}" style="height:${heightPct}%"></div>
        <div class="hcol-label">${h.hour}h</div>
      </div>`;
  }).join('');
}

/** Actualiza el badge del número de personas en el sidebar nav */
function updateSidebarBadge() {
  const badge = document.getElementById('badge-inside');
  if (badge) badge.textContent = INSIDE_NOW.length;
}


/* ================================================================
   5. NAVEGACIÓN ENTRE VISTAS
   Cada sección es un div.view; solo uno tiene class="active" a la vez.
================================================================ */

/* Metadatos de cada vista para actualizar el header */
const viewMeta = {
  dashboard: { title: 'Resumen del laboratorio',    sub: 'Vista general en tiempo real' },
  inside:    { title: 'Personas en el laboratorio', sub: 'Presencia actual en tiempo real' },
  access:    { title: 'Registro de accesos',        sub: 'Historial de entradas y salidas' },
  stats:     { title: 'Estadísticas de uso',        sub: 'Análisis de visitas y tendencias' },
  profile:   { title: 'Mi perfil',                  sub: 'Información del responsable del laboratorio' },
  settings:  { title: 'Configuración',              sub: 'Preferencias del sistema' }
};

/**
 * Navega a una vista.
 * @param {string} viewId  - ID de la vista (ej. 'dashboard', 'access')
 * @param {Element} navEl  - El elemento <a> del nav que fue clickeado
 */
function navigate(viewId, navEl) {
  // Ocultar todas las vistas
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));

  // Mostrar la vista seleccionada
  const target = document.getElementById(`view-${viewId}`);
  if (target) target.classList.add('active');

  // Resaltar el item del nav activo
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  if (navEl) navEl.classList.add('active');

  // Actualizar título y subtítulo del header
  const meta = viewMeta[viewId] || {};
  setEl('header-title', meta.title || viewId);
  setEl('header-sub',   meta.sub   || '');

  // Cerrar sidebar en móvil al navegar
  closeSidebar();
}


/* ================================================================
   6. SIDEBAR
   Móvil (<900px): se desliza con class .open en el sidebar.
   Escritorio: se colapsa con class .sidebar-collapsed en body,
   el CSS ajusta automáticamente el margen del .main.
================================================================ */

/** Abre o cierra el sidebar según el tamaño de pantalla */
function toggleSidebar() {
  if (window.innerWidth <= 900) {
    // Móvil: deslizar sidebar y mostrar overlay
    document.getElementById('sidebar').classList.toggle('open');
    document.getElementById('overlay').classList.toggle('active');
  } else {
    // Escritorio: colapsar/expandir con clase en body
    document.body.classList.toggle('sidebar-collapsed');
  }
}

/** Cierra el sidebar en móvil (lo llama el overlay y navigate) */
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('overlay').classList.remove('active');
}


/* ================================================================
   7. MODALES
================================================================ */

/** Abre un modal por su ID */
function showModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('open');
}

/** Cierra un modal por su ID */
function closeModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('open');
}

/** Cierra el modal si se hace clic en el overlay (fuera del modal) */
function closeModalOutside(event, id) {
  if (event.target.id === id) closeModal(id);
}


/* ================================================================
   8. PANEL DE NOTIFICACIONES (campana del header)
================================================================ */

/** Muestra u oculta el panel lateral de alertas */
function toggleNotifPanel() {
  document.getElementById('notifPanel').classList.toggle('open');
}


/* ================================================================
   9. RELOJ Y FECHA EN TIEMPO REAL
================================================================ */

/** Actualiza el reloj HH:MM:SS cada segundo */
function updateClock() {
  const now = new Date();
  const hh  = String(now.getHours()).padStart(2, '0');
  const mm  = String(now.getMinutes()).padStart(2, '0');
  const ss  = String(now.getSeconds()).padStart(2, '0');
  setEl('lab-clock', `${hh}:${mm}:${ss}`);
}

/** Escribe la fecha actual en el header (ej. "mié. 18 mar. 2026") */
function updateHeaderDate() {
  const d    = new Date();
  const opts = { weekday: 'short', day: '2-digit', month: 'short', year: 'numeric' };
  setEl('header-date', d.toLocaleDateString('es-MX', opts));
}


/* ================================================================
   10. MODO OSCURO / CLARO
   El toggle en Configuración llama a esta función.
================================================================ */

/** Alterna entre tema oscuro (default) y claro */
function toggleDark(isDark) {
  // isDark=true → quitar class light (mantener oscuro)
  // isDark=false → agregar class light
  document.body.classList.toggle('light', !isDark);
}


/* ================================================================
   11. ACTUALIZACIÓN AUTOMÁTICA DE DATOS
   En producción: reemplaza el cuerpo de refreshLiveData()
   con una llamada fetch() a tu API backend.
================================================================ */

/**
 * Refresca los datos en vivo.
 * Actualmente simula el refresh. En producción:
 *   const data = await fetch('/api/lab/inside').then(r => r.json());
 *   // actualizar INSIDE_NOW con data, luego llamar render...
 */
function refreshLiveData() {
  updateKPIs();
  renderInsideListDash();
  renderInsideTable();
  updateSidebarBadge();
  checkAlerts();
}


/* ================================================================
   12. INICIALIZACIÓN
   Se ejecuta cuando el DOM está listo.
================================================================ */
document.addEventListener('DOMContentLoaded', () => {
  // Arrancar reloj y fecha
  updateClock();
  updateHeaderDate();
  setInterval(updateClock, 1000);

  // Pintar todo el dashboard
  renderAll();

  // Primera revisión de alertas: actualiza contadores y listas
  // pero NO dispara modal/toast (marcar alertas actuales como ya vistas)
  const alertsAtLoad = getActiveAlerts();
  alertsAtLoad.forEach(p => notifiedIds.add(p.id));  // silenciar alertas previas al abrir
  checkAlerts();

  // Revisión periódica: a partir de aquí SÍ se disparan modales para nuevas alertas
  setInterval(checkAlerts, 60_000);

  // Actualizar barras de tiempo y tabla de dentro cada 30s
  setInterval(() => {
    renderInsideListDash();
    renderInsideTable();
  }, 30_000);

  // Refresco general de datos cada 30s
  setInterval(refreshLiveData, 30_000);

  // Conectar botón de confirmar cierre de sesión
  const btnConfirmLogout = document.getElementById('btn-confirm-logout');
  if (btnConfirmLogout) {
    btnConfirmLogout.addEventListener('click', () => {
      // Aquí puedes agregar lógica de logout real (limpiar sesión, redirigir, etc.)
      // Por ejemplo: window.location.href = '/login';
      document.body.innerHTML = `
        <div style="
          display:flex; flex-direction:column; align-items:center; justify-content:center;
          min-height:100vh; background:var(--bg); font-family:var(--font-ui); gap:16px;
        ">
          <div style="
            width:64px; height:64px; border-radius:50%;
            background:var(--accent-dim); border:2px solid var(--accent);
            display:flex; align-items:center; justify-content:center;
          ">
            <svg viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2" width="28" height="28">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
              <polyline points="16 17 21 12 16 7"/>
              <line x1="21" y1="12" x2="9" y2="12"/>
            </svg>
          </div>
          <p style="font-size:18px; font-weight:600; color:var(--text-1)">Sesión cerrada</p>
          <p style="font-size:13px; color:var(--text-3)">Has salido de LabControl correctamente.</p>
          <button onclick="location.reload()" style="
            margin-top:8px; padding:10px 24px;
            background:var(--accent); color:#0F1117;
            border:none; border-radius:8px; font-size:13px; font-weight:700;
            cursor:pointer; font-family:var(--font-ui);
          ">Volver al inicio</button>
        </div>`;
    });
  }
});