/* ═══════════════════════════════════════════════════════
   alerts.js — Sistema de alertas de permanencia.
═══════════════════════════════════════════════════════ */

/* ================================================================
   3. SISTEMA DE ALERTAS DE PERMANENCIA
   Detecta personas que llevan más tiempo del límite configurado
   y muestra modal + toast + badge en la campana.
================================================================ */

let TIME_LIMIT_MINUTES = 120;         // límite default: 2 horas
const notifiedIds      = new Set();   // IDs ya notificados (evita spam de modales)

/* ── 3a. Revisar alertas — se llama periódicamente ── */
function checkAlerts() {
  const alerts = getActiveAlerts();
  const count  = alerts.length;

  // Actualizar número en KPI de alertas
  setEl('kpi-alerts', count);

  // Mostrar/ocultar contador rojo en campana
  const counter = document.getElementById('notif-counter');
  if (counter) {
    counter.textContent    = count;
    counter.style.display  = count > 0 ? 'flex' : 'none';
  }

  // Actualizar lista en el card del dashboard
  renderAlertsList(alerts);

  // Actualizar panel lateral de la campana
  renderNotifPanel(alerts);

  // Disparar modal + toast solo para personas que no han sido notificadas aún
  alerts.forEach(person => {
    if (!notifiedIds.has(person.id)) {
      notifiedIds.add(person.id);
      showTimeAlertModal(person);
      showToast(person);
    }
  });
}

/** Filtra INSIDE_NOW y devuelve solo quienes superaron el límite */
function getActiveAlerts() {
  return INSIDE_NOW.filter(p => minutesElapsed(p.entryTime) >= TIME_LIMIT_MINUTES);
}

/* ── 3b. Renderizado de la lista de alertas en el card del dashboard ── */
function renderAlertsList(alerts) {
  const container = document.getElementById('alerts-list');
  if (!container) return;

  if (alerts.length === 0) {
    container.innerHTML = `
      <div class="empty-state-sm">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="28" height="28">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
          <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        <span>Sin alertas activas</span>
      </div>`;
    return;
  }

  container.innerHTML = alerts.map(p => {
    const mins = minutesElapsed(p.entryTime);
    return `
      <div class="alert-item" id="alert-${p.id}">
        <div class="alert-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
            <circle cx="12" cy="12" r="10"/>
            <polyline points="12 6 12 12 16 14"/>
          </svg>
        </div>
        <div class="alert-info">
          <div class="alert-name">${p.name}</div>
          <div class="alert-elapsed">Lleva ${formatDuration(mins)} · Límite: ${formatDuration(TIME_LIMIT_MINUTES)}</div>
        </div>
        <button class="alert-dismiss" onclick="dismissAlert('${p.id}')" title="Descartar">✕</button>
      </div>`;
  }).join('');
}

/** Renderiza el panel lateral de la campana con las alertas activas */
function renderNotifPanel(alerts) {
  const body = document.getElementById('notifPanelBody');
  if (!body) return;

  if (alerts.length === 0) {
    body.innerHTML = '<div class="notif-empty">Sin alertas activas</div>';
    return;
  }

  body.innerHTML = alerts.map(p => {
    const mins = minutesElapsed(p.entryTime);
    return `
      <div class="notif-item">
        <div class="notif-item-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        </div>
        <div class="notif-item-body">
          <div class="notif-item-name">${p.name}</div>
          <div class="notif-item-time">Dentro hace ${formatDuration(mins)} — límite superado</div>
        </div>
      </div>`;
  }).join('');
}

/* ── 3c. Modal emergente y Toast de alerta ── */

/** Muestra el modal de alerta con el nombre y tiempo de la persona */
function showTimeAlertModal(person) {
  const mins = minutesElapsed(person.entryTime);
  const desc = document.getElementById('timeAlertDesc');
  if (desc) {
    desc.textContent =
      `${person.name} lleva ${formatDuration(mins)} en el laboratorio. ` +
      `El límite configurado es de ${formatDuration(TIME_LIMIT_MINUTES)}. ` +
      `Por favor verifica su situación.`;
  }
  showModal('timeAlertModal');
}

/** Muestra un toast (notificación esquina) con auto-cierre a los 8 segundos */
function showToast(person) {
  // Crear contenedor si no existe
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const mins = minutesElapsed(person.entryTime);
  const id   = `toast-${person.id}-${Date.now()}`;

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.id = id;
  toast.innerHTML = `
    <div class="toast-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
    </div>
    <div class="toast-body">
      <div class="toast-title">Alerta de permanencia</div>
      <div class="toast-msg">${person.name} lleva ${formatDuration(mins)} — límite superado</div>
    </div>
    <button class="toast-close" onclick="removeToast('${id}')">✕</button>`;

  container.appendChild(toast);
  setTimeout(() => removeToast(id), 8000);  // auto-cierre
}

/** Elimina un toast del DOM */
function removeToast(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

/** Descarta manualmente una alerta del card (sin afectar la lista general) */
function dismissAlert(personId) {
  notifiedIds.delete(personId);          // permite re-notificar si sigue dentro
  const el = document.getElementById(`alert-${personId}`);
  if (el) el.remove();
  checkAlerts();
}

/* ── 3d. Configurar límite de tiempo ── */

/** Actualiza el límite de permanencia y recalcula alertas activas */
function updateTimeLimit(value) {
  TIME_LIMIT_MINUTES = parseInt(value, 10);
  notifiedIds.clear();  // resetear notificados para recalcular con el nuevo límite

  // Actualizar etiqueta visible en el card
  const label = document.getElementById('limit-label');
  if (label) label.textContent = formatDuration(TIME_LIMIT_MINUTES);

  // Sincronizar el select de configuración
  const sel = document.getElementById('time-limit-select');
  if (sel) sel.value = value;

  checkAlerts();
}

/** Aplica el límite seleccionado en el modal de configuración */
function applyTimeLimit() {
  const val = document.getElementById('modal-time-select').value;
  updateTimeLimit(val);
  closeModal('timeConfigModal');
}