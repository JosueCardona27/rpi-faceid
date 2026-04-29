/* ═══════════════════════════════════════════════════════
   app.js — Lógica principal del dashboard.
═══════════════════════════════════════════════════════ */

/* ── Guard de sesión ─────────────────────────────────────────────────────────
   Si no hay sesión activa redirige al login.
   Solo pueden acceder maestros y administradores.
────────────────────────────────────────────────────────────────────────────── */
(function checkSession() {
  const raw = sessionStorage.getItem('usuario_login');
  if (!raw) {
    window.location.href = '../login/login.html';
    return;
  }
  const u = JSON.parse(raw);
  if (u.rol !== 'admin' && u.rol !== 'maestro') {
    sessionStorage.removeItem('usuario_login');
    window.location.href = '../login/login.html';
  }
})();

const _SESION   = JSON.parse(sessionStorage.getItem('usuario_login') || '{}');
const _ES_ADMIN = _SESION.rol === 'admin';


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

  if (typeF)   rows = rows.filter(r => r.type   === typeF);
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
================================================================ */

const viewMeta = {
  dashboard: { title: 'Resumen del laboratorio',    sub: 'Vista general en tiempo real' },
  inside:    { title: 'Personas en el laboratorio', sub: 'Presencia actual en tiempo real' },
  access:    { title: 'Registro de accesos',        sub: 'Historial de entradas y salidas' },
  stats:     { title: 'Estadísticas de uso',        sub: 'Análisis de visitas y tendencias' },
  profile:   { title: 'Mi perfil',                  sub: 'Información del responsable del laboratorio' },
  settings:  { title: 'Configuración',              sub: 'Preferencias del sistema' },
  // vistas de gestión
  students:  { title: 'Gestión de Alumnos',         sub: 'Registra, edita y elimina alumnos' },
  teachers:  { title: 'Gestión de Maestros',        sub: 'Administra los maestros del sistema' },
  admins:    { title: 'Gestión de Administradores', sub: 'Administra los administradores del sistema' }
};

function navigate(viewId, navEl) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));

  const target = document.getElementById(`view-${viewId}`);
  if (target) target.classList.add('active');

  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  if (navEl) navEl.classList.add('active');

  const meta = viewMeta[viewId] || {};
  setEl('header-title', meta.title || viewId);
  setEl('header-sub',   meta.sub   || '');

  closeSidebar();
}


/* ================================================================
   6. SIDEBAR
================================================================ */

function toggleSidebar() {
  if (window.innerWidth <= 900) {
    document.getElementById('sidebar').classList.toggle('open');
    document.getElementById('overlay').classList.toggle('active');
  } else {
    document.body.classList.toggle('sidebar-collapsed');
  }
}

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('overlay').classList.remove('active');
}


/* ================================================================
   7. MODALES
================================================================ */

function showModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('open');
}

function closeModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('open');
}

function closeModalOutside(event, id) {
  if (event.target.id === id) closeModal(id);
}


/* ================================================================
   8. PANEL DE NOTIFICACIONES (campana del header)
================================================================ */

function toggleNotifPanel() {
  document.getElementById('notifPanel').classList.toggle('open');
}


/* ================================================================
   9. RELOJ Y FECHA EN TIEMPO REAL
================================================================ */

function updateClock() {
  const now = new Date();
  const hh  = String(now.getHours()).padStart(2, '0');
  const mm  = String(now.getMinutes()).padStart(2, '0');
  const ss  = String(now.getSeconds()).padStart(2, '0');
  setEl('lab-clock', `${hh}:${mm}:${ss}`);
}

function updateHeaderDate() {
  const d    = new Date();
  const opts = { weekday: 'short', day: '2-digit', month: 'short', year: 'numeric' };
  setEl('header-date', d.toLocaleDateString('es-MX', opts));
}


/* ================================================================
   10. MODO OSCURO / CLARO
================================================================ */

function toggleDark(isDark) {
  document.body.classList.toggle('light', !isDark);
}


/* ================================================================
   11. ACTUALIZACIÓN AUTOMÁTICA DE DATOS
================================================================ */

function refreshLiveData() {
  updateKPIs();
  renderInsideListDash();
  renderInsideTable();
  updateSidebarBadge();
  checkAlerts();
}


/* ================================================================
   12. INICIALIZACIÓN
================================================================ */
document.addEventListener('DOMContentLoaded', () => {

  // ── Perfil dinámico desde la sesión ──
  const initials = ((_SESION.nombre||'')[0] + (_SESION.apellido_paterno||'')[0]).toUpperCase();
  document.querySelectorAll('.profile-avatar-span').forEach(el => el.textContent = initials);
  setEl('profile-name-sidebar',  `${_SESION.nombre||''} ${_SESION.apellido_paterno||''}`);
  setEl('profile-role-sidebar',  _SESION.rol === 'admin' ? 'Administrador' : 'Maestro');
  setEl('profile-dept-sidebar',  _SESION.correo || _SESION.numero_cuenta || '');

  // Perfil de la vista "Mi perfil"
  setEl('page-profile-initials', initials);
  setEl('page-profile-name',     `${_SESION.nombre||''} ${_SESION.apellido_paterno||''} ${_SESION.apellido_materno||''}`);
  setEl('page-profile-role',     _SESION.rol === 'admin' ? 'Administrador' : 'Maestro');
  setEl('page-profile-correo',   _SESION.correo        || '—');
  setEl('page-profile-cuenta',   _SESION.numero_cuenta || '—');

  // ── Mostrar/ocultar secciones de admin ──
  if (!_ES_ADMIN) {
    document.querySelectorAll('.admin-only').forEach(el => el.style.display = 'none');
  }

  // ── Cargar tablas de gestión desde la BD ──
  crudLoadAll().catch(e => console.warn('CRUD init:', e.message));

  // ── Reloj y fecha ──
  updateClock();
  updateHeaderDate();
  setInterval(updateClock, 1000);

  // ── Renderizado del dashboard ──
  renderAll();

  // ── Alertas ──
  const alertsAtLoad = getActiveAlerts();
  alertsAtLoad.forEach(p => notifiedIds.add(p.id));
  checkAlerts();

  setInterval(checkAlerts, 60_000);
  setInterval(() => { renderInsideListDash(); renderInsideTable(); }, 30_000);
  setInterval(refreshLiveData, 30_000);

  // ── Cerrar sesión ──
  const btnLogout = document.getElementById('btn-confirm-logout');
  if (btnLogout) {
    btnLogout.addEventListener('click', () => {
      sessionStorage.removeItem('usuario_login');
      window.location.href = '../login/login.html';
    });
  }
});
