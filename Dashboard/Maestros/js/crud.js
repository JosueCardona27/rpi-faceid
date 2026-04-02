/* ═══════════════════════════════════════════════════════
   crud.js — Gestión de personas + verificación facial
             contra el sistema Python (api.py).
═══════════════════════════════════════════════════════ */

'use strict';

// ── Config ──────────────────────────────────────────────────────────────────
const CRUD_DB_PATH = '../../database/reconocimiento_facial.db';

// URL de la API Python (api.py corriendo con Flask).
// Si el navegador y el servidor Python están en la misma máquina, es localhost.
const API_URL = 'http://localhost:5050';

// ── Estado interno ──────────────────────────────────────────────────────────
let _crudSQL  = null;
let _crudDB   = null;
let _editId   = null;   // null = nuevo, número = editar
let _editRole = null;
let _delId    = null;
let _delName  = '';

// Cámara
let _videoStream  = null;
let _captureTimer = null;   // intervalo de captura de frames a la API

// ── Sesión activa ────────────────────────────────────────────────────────────
function _getSession() {
  const raw = sessionStorage.getItem('usuario_login');
  return raw ? JSON.parse(raw) : null;
}

// ── SHA-256 (igual que Python hashlib) ───────────────────────────────────────
async function _sha256(text) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2,'0')).join('');
}

// ── Datetime local ───────────────────────────────────────────────────────────
function _now() {
  const d = new Date();
  const p = n => String(n).padStart(2,'0');
  return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())} ` +
         `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

// ── Verificar si la API Python está activa ────────────────────────────────────
async function _apiActiva() {
  try {
    const r = await fetch(`${API_URL}/api/ping`, { signal: AbortSignal.timeout(2000) });
    return r.ok;
  } catch { return false; }
}

// ── Inicializar BD (sql.js) ──────────────────────────────────────────────────
async function _ensureDB() {
  if (_crudDB) return _crudDB;

  if (!_crudSQL) {
    _crudSQL = await initSqlJs({
      locateFile: f => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.3/${f}`
    });
  }

  const resp = await fetch(CRUD_DB_PATH + '?t=' + Date.now());
  if (!resp.ok) throw new Error(`No se pudo cargar la BD (HTTP ${resp.status})`);
  const buf = await resp.arrayBuffer();
  _crudDB   = new _crudSQL.Database(new Uint8Array(buf));
  return _crudDB;
}

// ── Ejecutar SELECT → array de objetos ──────────────────────────────────────
function _query(sql, params = []) {
  const res = _crudDB.exec(sql, params);
  if (!res || res.length === 0) return [];
  const { columns, values } = res[0];
  return values.map(row => {
    const obj = {};
    columns.forEach((c, i) => obj[c] = row[i]);
    return obj;
  });
}

// ── Ejecutar INSERT / UPDATE / DELETE ────────────────────────────────────────
function _run(sql, params = []) { _crudDB.run(sql, params); }

// ── Último id insertado ──────────────────────────────────────────────────────
function _lastId() {
  const r = _crudDB.exec('SELECT last_insert_rowid()');
  return r[0]?.values[0]?.[0] ?? null;
}

// ── Exportar BD actualizada ──────────────────────────────────────────────────
function _exportDB() {
  if (!_crudDB) return;
  const blob = new Blob([_crudDB.export()], { type: 'application/octet-stream' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'reconocimiento_facial.db';
  a.click();
}


/* ══════════════════════════════════════════════════════
   CARGAR DATOS Y RENDERIZAR TABLAS
══════════════════════════════════════════════════════ */

async function crudLoadAll() {
  await _ensureDB();
  _verificarColumnas();
  crudRenderStudents();
  const s = _getSession();
  if (s && s.rol === 'admin') {
    crudRenderTeachers();
    crudRenderAdmins();
  }
}

/**
 * Verifica si las columnas nuevas existen.
 * Si no, las agrega en memoria (no modifica el archivo .db en disco).
 * Para persistir hay que descargar la BD después.
 */
function _verificarColumnas() {
  try {
    _crudDB.exec('SELECT fecha_actualizacion, actualizado_por FROM usuarios LIMIT 1');
  } catch {
    // Las columnas no existen todavía — agregarlas en esta sesión
    try {
      _crudDB.run('ALTER TABLE usuarios ADD COLUMN fecha_actualizacion TEXT');
      _crudDB.run('ALTER TABLE usuarios ADD COLUMN actualizado_por INTEGER');
      console.info('[CRUD] Columnas fecha_actualizacion y actualizado_por agregadas en memoria.');
      _showExportBanner('Las columnas nuevas se agregaron en memoria. Descarga la BD para guardarlas permanentemente.');
    } catch (e2) {
      console.warn('[CRUD] No se pudieron agregar columnas:', e2.message);
    }
  }
}

// ── Helper: fecha legible ────────────────────────────────────────────────────
function _fmt(str) {
  if (!str) return '—';
  const d = new Date(str.replace(' ','T'));
  if (isNaN(d)) return str;
  const p = n => String(n).padStart(2,'0');
  return `${p(d.getDate())}/${p(d.getMonth()+1)}/${d.getFullYear()} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function _avatar(u) {
  return ((u.nombre||' ')[0] + (u.apellido_paterno||' ')[0]).toUpperCase();
}

function esc(s) { return (s||'').replace(/'/g,"\\'").replace(/"/g,'&quot;'); }

// ── Consulta genérica por rol ────────────────────────────────────────────────
function _rows(role) {
  // Consulta adaptable: usa LEFT JOIN en actualizado_por solo si la columna existe
  try {
    return _query(`
      SELECT u.id,
             u.nombre, u.apellido_paterno, u.apellido_materno,
             u.numero_cuenta, u.correo, u.rol,
             u.fecha_registro,
             u.fecha_actualizacion,
             ed.grado, ed.grupo,
             ua.nombre || ' ' || ua.apellido_paterno AS actualizado_por_nombre
      FROM usuarios u
      LEFT JOIN estudiantes_detalle ed ON ed.usuario_id = u.id
      LEFT JOIN usuarios ua ON ua.id = u.actualizado_por
      WHERE u.rol = ?
      ORDER BY u.apellido_paterno, u.nombre
    `, [role]);
  } catch {
    // Fallback sin las columnas nuevas
    return _query(`
      SELECT u.id,
             u.nombre, u.apellido_paterno, u.apellido_materno,
             u.numero_cuenta, u.correo, u.rol,
             u.fecha_registro,
             NULL AS fecha_actualizacion,
             ed.grado, ed.grupo,
             NULL AS actualizado_por_nombre
      FROM usuarios u
      LEFT JOIN estudiantes_detalle ed ON ed.usuario_id = u.id
      WHERE u.rol = ?
      ORDER BY u.apellido_paterno, u.nombre
    `, [role]);
  }
}

// ── Renderizar Alumnos ────────────────────────────────────────────────────────
function crudRenderStudents() {
  const tbody = document.getElementById('students-tbody');
  if (!tbody || !_crudDB) return;
  const rows = _rows('estudiante');

  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="table-empty">No hay alumnos registrados.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(u => `
    <tr>
      <td class="td-name">
        <div class="person-cell">
          <div class="mini-avatar">${_avatar(u)}</div>
          ${u.nombre} ${u.apellido_paterno} ${u.apellido_materno||''}
        </div>
      </td>
      <td class="td-mono">${u.numero_cuenta||'—'}</td>
      <td>${u.grado ? `${u.grado}° ${u.grupo||''}` : '—'}</td>
      <td class="td-mono td-small">${_fmt(u.fecha_registro)}</td>
      <td class="td-small">${u.actualizado_por_nombre||'—'}</td>
      <td class="td-mono td-small">${_fmt(u.fecha_actualizacion)}</td>
      <td>
        <div class="action-btns">
          <button class="action-btn edit"   onclick="crudOpenEdit(${u.id},'estudiante')" title="Editar">✏️</button>
          <button class="action-btn remove" onclick="crudOpenDelete(${u.id},'${esc(u.nombre+' '+u.apellido_paterno)}')" title="Eliminar">🗑️</button>
        </div>
      </td>
    </tr>`).join('');
}

// ── Renderizar Maestros ───────────────────────────────────────────────────────
function crudRenderTeachers() {
  const tbody = document.getElementById('teachers-tbody');
  if (!tbody || !_crudDB) return;
  const rows = _rows('maestro');

  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="table-empty">No hay maestros registrados.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(u => `
    <tr>
      <td class="td-name">
        <div class="person-cell">
          <div class="mini-avatar">${_avatar(u)}</div>
          ${u.nombre} ${u.apellido_paterno} ${u.apellido_materno||''}
        </div>
      </td>
      <td class="td-mono">${u.numero_cuenta||'—'}</td>
      <td class="td-small">${u.correo||'—'}</td>
      <td class="td-mono td-small">${_fmt(u.fecha_registro)}</td>
      <td class="td-small">${u.actualizado_por_nombre||'—'}</td>
      <td class="td-mono td-small">${_fmt(u.fecha_actualizacion)}</td>
      <td>
        <div class="action-btns">
          <button class="action-btn edit"   onclick="crudOpenEdit(${u.id},'maestro')" title="Editar">✏️</button>
          <button class="action-btn remove" onclick="crudOpenDelete(${u.id},'${esc(u.nombre+' '+u.apellido_paterno)}')" title="Eliminar">🗑️</button>
        </div>
      </td>
    </tr>`).join('');
}

// ── Renderizar Admins ─────────────────────────────────────────────────────────
function crudRenderAdmins() {
  const tbody = document.getElementById('admins-tbody');
  if (!tbody || !_crudDB) return;
  const s    = _getSession();
  const rows = _rows('admin');

  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="table-empty">No hay administradores registrados.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(u => {
    const esSelf = s && u.id === s.id;
    return `
    <tr ${esSelf ? 'class="self-row"' : ''}>
      <td class="td-name">
        <div class="person-cell">
          <div class="mini-avatar ${esSelf ? 'self' : ''}">${_avatar(u)}</div>
          ${u.nombre} ${u.apellido_paterno} ${u.apellido_materno||''}
          ${esSelf ? '<span class="self-badge">Tú</span>' : ''}
        </div>
      </td>
      <td class="td-mono">${u.numero_cuenta||'—'}</td>
      <td class="td-small">${u.correo||'—'}</td>
      <td class="td-mono td-small">${_fmt(u.fecha_registro)}</td>
      <td class="td-small">${u.actualizado_por_nombre||'—'}</td>
      <td>
        <div class="action-btns">
          <button class="action-btn edit"   onclick="crudOpenEdit(${u.id},'admin')" title="Editar">✏️</button>
          <button class="action-btn remove" onclick="crudOpenDelete(${u.id},'${esc(u.nombre+' '+u.apellido_paterno)}')"
            ${esSelf ? 'disabled title="No puedes eliminarte a ti mismo"' : 'title="Eliminar"'}>🗑️</button>
        </div>
      </td>
    </tr>`;
  }).join('');
}


/* ══════════════════════════════════════════════════════
   ABRIR MODALES (CRUD)
══════════════════════════════════════════════════════ */

// ── Nuevo: primero captura numero_cuenta, luego verificación facial ───────────
async function crudOpenAdd(role) {
  _editId   = null;
  _editRole = role;

  // Antes de abrir el modal facial, pedir el número de cuenta para saber
  // si la persona ya tiene vectores registrados (y si hay que verificar).
  const cuenta = await _pedirCuenta(role);
  if (cuenta === null) return;   // canceló

  // Pre-llenar el campo de cuenta en el formulario
  document.getElementById('fm-cuenta').value = cuenta;

  // Verificar si la API Python está activa y si la persona tiene vectores
  const apiOk  = await _apiActiva();
  const status = document.getElementById('face-status');
  const btnConf = document.getElementById('btn-face-confirm');

  const roleLabel = { estudiante:'alumno', maestro:'maestro', admin:'administrador' };
  document.getElementById('faceModalTitle').textContent =
    `Verificar identidad — nuevo ${roleLabel[role]||role}`;

  btnConf.disabled    = true;
  btnConf.textContent = 'Continuar al registro →';
  status.textContent  = 'Iniciando cámara…';
  status.className    = 'face-status waiting';
  document.getElementById('face-canvas').style.display = 'none';

  _showModal('faceModal');

  if (!apiOk) {
    // API no disponible — solo detectar presencia de cara (comportamiento anterior)
    _setFaceNote('⚠ API Python no disponible. Solo se verificará presencia de cara.', 'warn');
    _startCameraBasic(btnConf, status);
    return;
  }

  // API activa — verificar si tiene vectores
  let tieneVectores  = false;
  let nombreRegistro = null;
  try {
    const r = await fetch(`${API_URL}/api/tiene_vectores/${encodeURIComponent(cuenta)}`);
    const d = await r.json();
    tieneVectores  = d.tiene_vectores;
    nombreRegistro = d.nombre;
  } catch {
    tieneVectores = false;
  }

  if (!tieneVectores) {
    // No tiene vectores → no hay contra qué comparar
    _setFaceNote(
      '⚠ Esta persona aún no tiene rostro registrado en el terminal biométrico.<br>' +
      'Puedes continuar el registro de datos, pero deberá registrarse físicamente después.',
      'warn'
    );
    status.textContent = 'Sin vectores almacenados — solo presencia de cara.';
    status.className   = 'face-status waiting';
    _startCameraBasic(btnConf, status);
    return;
  }

  // Tiene vectores → verificación real contra el sistema Python
  _setFaceNote(`Verificando que sea <strong>${nombreRegistro}</strong>…`, 'info');
  status.textContent = 'Buscando rostro…';
  status.className   = 'face-status waiting';
  _startCameraVerify(cuenta, nombreRegistro, btnConf, status);
}

// ── Pedir número de cuenta antes de abrir el modal facial ────────────────────
function _pedirCuenta(role) {
  return new Promise(resolve => {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay open';
    overlay.id = '_cuenta-prompt';

    const roleLabel = { estudiante:'del alumno', maestro:'del maestro', admin:'del administrador' };

    overlay.innerHTML = `
      <div class="modal" style="max-width:380px">
        <div class="modal-title">Número de cuenta</div>
        <div class="modal-desc">
          Ingresa el número de cuenta ${roleLabel[role]||''} antes de iniciar la verificación facial.
        </div>
        <div class="fm-field" style="margin-bottom:0">
          <label class="fm-label">Número de cuenta (8 dígitos)</label>
          <input class="fm-input" type="text" id="_prompt-cuenta"
            maxlength="8" placeholder="Ej. 20220001" autofocus>
        </div>
        <div id="_prompt-err" style="font-size:11px;color:var(--red);margin-top:6px;min-height:16px"></div>
        <div class="modal-actions" style="margin-top:14px">
          <button class="btn-cancel" id="_prompt-cancel">Cancelar</button>
          <button class="btn-confirm info" id="_prompt-ok">Continuar</button>
        </div>
      </div>`;

    document.body.appendChild(overlay);

    const input  = overlay.querySelector('#_prompt-cuenta');
    const errEl  = overlay.querySelector('#_prompt-err');
    const ok     = overlay.querySelector('#_prompt-ok');
    const cancel = overlay.querySelector('#_prompt-cancel');

    function _close(val) {
      overlay.remove();
      resolve(val);
    }

    function _submit() {
      const v = input.value.trim();
      if (!/^\d{8}$/.test(v)) {
        errEl.textContent = 'Debe tener exactamente 8 dígitos.';
        input.focus();
        return;
      }
      _close(v);
    }

    ok.onclick     = _submit;
    cancel.onclick = () => _close(null);
    input.addEventListener('keydown', e => { if (e.key === 'Enter') _submit(); });
    input.focus();
  });
}

// ── Nota informativa dentro del modal facial ─────────────────────────────────
function _setFaceNote(html, type) {
  let note = document.getElementById('face-note');
  if (!note) {
    note = document.createElement('div');
    note.id = 'face-note';
    note.style.cssText = 'font-size:12px;padding:7px 10px;border-radius:6px;margin-bottom:10px;text-align:center';
    document.querySelector('#faceModal .modal-desc')?.after(note);
  }
  const colors = {
    info: 'var(--blue-dim)',
    warn: 'var(--amber-dim)',
    ok:   'var(--accent-dim)',
    err:  'var(--red-dim)',
  };
  const textColors = { info:'var(--blue)', warn:'var(--amber)', ok:'var(--accent)', err:'var(--red)' };
  note.style.background = colors[type] || colors.info;
  note.style.color      = textColors[type] || textColors.info;
  note.innerHTML        = html;
}

// ── Cámara básica: solo detecta presencia de cara (sin API Python) ────────────
async function _startCameraBasic(btnConf, status) {
  try {
    _videoStream = await navigator.mediaDevices.getUserMedia({ video: { width:320, height:240 } });
    const video  = document.getElementById('face-video');
    video.srcObject = _videoStream;
    await video.play();

    await faceapi.nets.tinyFaceDetector.loadFromUri(
      'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model'
    );

    _captureTimer = setInterval(async () => {
      if (!video.srcObject) return;
      const dets = await faceapi.detectAllFaces(
        video, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.5 })
      );
      if (dets.length > 0) {
        status.textContent = '✓ Cara detectada — puedes continuar';
        status.className   = 'face-status ok';
        btnConf.disabled   = false;
        _drawFaceBox(video, dets[0].box, '#00C9A7');
      } else {
        status.textContent = 'Buscando cara…';
        status.className   = 'face-status waiting';
        btnConf.disabled   = true;
        document.getElementById('face-canvas').style.display = 'none';
      }
    }, 600);

  } catch (e) {
    status.textContent = '✗ Sin cámara: ' + e.message;
    status.className   = 'face-status error';
    btnConf.disabled   = false;
    btnConf.textContent = 'Continuar sin cámara';
  }
}

// ── Cámara con verificación real contra la API Python ────────────────────────
async function _startCameraVerify(cuenta, nombreEsperado, btnConf, status) {
  const UMBRAL_SIM = 60;   // similitud mínima en % para aprobar

  try {
    _videoStream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480 } });
    const video  = document.getElementById('face-video');
    video.srcObject = _videoStream;
    await video.play();

    let verificando = false;

    _captureTimer = setInterval(async () => {
      if (!video.srcObject || verificando) return;
      verificando = true;

      try {
        // Capturar frame del video como base64
        const canvas  = document.createElement('canvas');
        canvas.width  = video.videoWidth  || 640;
        canvas.height = video.videoHeight || 480;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const b64 = canvas.toDataURL('image/jpeg', 0.8);

        const resp = await fetch(`${API_URL}/api/reconocer_cuenta`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ imagen: b64, numero_cuenta: cuenta }),
          signal:  AbortSignal.timeout(5000),
        });

        const data = await resp.json();

        if (!data.ok) {
          status.textContent = data.error || 'Error en la API';
          status.className   = 'face-status error';
          btnConf.disabled   = true;
          _setFaceNote('✗ ' + (data.error || 'Error desconocido'), 'err');

        } else if (!data.tiene_rostro && data.coincide === undefined) {
          // Sin rostro detectado por Python
          status.textContent = 'Acércate a la cámara…';
          status.className   = 'face-status waiting';
          btnConf.disabled   = true;
          _drawEmpty();

        } else if (data.sin_vectores) {
          // Sin vectores (no debería llegar aquí, pero por si acaso)
          status.textContent = 'Sin vectores registrados';
          status.className   = 'face-status waiting';
          btnConf.disabled   = false;

        } else if (data.coincide && data.similitud_pct >= UMBRAL_SIM) {
          // ✓ Verificado
          const sim = data.similitud_pct;
          status.textContent = `✓ Identidad confirmada — ${nombreEsperado} (${sim}%)`;
          status.className   = 'face-status ok';
          btnConf.disabled   = false;
          _setFaceNote(`✓ Coincide con <strong>${data.nombre}</strong> — similitud ${sim}%`, 'ok');
          _drawFaceBox(video, null, '#00C9A7', `${sim}%`);

        } else {
          // No coincide o similitud baja
          const sim  = data.similitud_pct ?? 0;
          const msg  = data.coincide === false
            ? `No coincide con ${nombreEsperado} (${sim}%)`
            : `Similitud baja: ${sim}%`;
          status.textContent = msg;
          status.className   = 'face-status error';
          btnConf.disabled   = true;
          _setFaceNote(`✗ ${msg}. Asegúrate de estar mirando de frente.`, 'err');
          _drawFaceBox(video, null, '#EF4444', `${sim}%`);
        }

      } catch (e) {
        status.textContent = 'Error conectando a API…';
        status.className   = 'face-status waiting';
        btnConf.disabled   = true;
      } finally {
        verificando = false;
      }
    }, 1200);   // cada 1.2s para no saturar la API

  } catch (e) {
    status.textContent = '✗ Sin cámara: ' + e.message;
    status.className   = 'face-status error';
    btnConf.disabled   = false;
    btnConf.textContent = 'Continuar sin cámara';
  }
}

// ── Dibujar caja en el canvas de la vista facial ─────────────────────────────
function _drawFaceBox(video, box, color, label) {
  const canvas  = document.getElementById('face-canvas');
  canvas.width  = video.videoWidth  || 320;
  canvas.height = video.videoHeight || 240;
  canvas.style.display = 'block';
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  if (box) {
    ctx.strokeStyle = color;
    ctx.lineWidth   = 3;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
  }
  if (label) {
    ctx.fillStyle = color;
    ctx.font      = 'bold 16px monospace';
    ctx.fillText(label, 12, 24);
  }
}

function _drawEmpty() {
  const canvas = document.getElementById('face-canvas');
  canvas.style.display = 'none';
}

// ── Editar (sin verificación facial — ya está registrado) ─────────────────────
function crudOpenEdit(id, role) {
  _editId   = id;
  _editRole = role;

  const rows = _query('SELECT * FROM usuarios WHERE id = ?', [id]);
  if (!rows.length) return;
  const u = rows[0];

  let grado = '', grupo = '';
  if (role === 'estudiante') {
    const det = _query('SELECT grado, grupo FROM estudiantes_detalle WHERE usuario_id = ?', [id]);
    if (det.length) { grado = det[0].grado||''; grupo = det[0].grupo||''; }
  }

  const titles = { estudiante:'Editar alumno', maestro:'Editar maestro', admin:'Editar administrador' };
  document.getElementById('personModalTitle').textContent = titles[role]||'Editar';

  document.getElementById('fm-nombre').value = u.nombre       || '';
  document.getElementById('fm-ap').value     = u.apellido_paterno || '';
  document.getElementById('fm-am').value     = (u.apellido_materno === '.' ? '' : u.apellido_materno||'');
  document.getElementById('fm-cuenta').value = u.numero_cuenta || '';
  document.getElementById('fm-correo').value = u.correo       || '';
  document.getElementById('fm-grado').value  = grado;
  document.getElementById('fm-grupo').value  = grupo;
  document.getElementById('fm-pwd').value    = '';
  document.getElementById('fm-pwd').placeholder = 'Dejar vacío para no cambiar';

  _configureForm(role);
  _showModal('personModal');
}

// ── Eliminar ──────────────────────────────────────────────────────────────────
function crudOpenDelete(id, name) {
  _delId   = id;
  _delName = name;
  document.getElementById('deleteModalDesc').textContent =
    `¿Seguro que deseas eliminar a ${name}? Esta acción no se puede deshacer.`;
  _showModal('deleteModal');
}

// ── El modal facial confirmó → abrir formulario ──────────────────────────────
function crudFaceConfirmed() {
  _stopCamera();
  _closeModal('faceModal');

  // Limpiar la nota del modal facial para la próxima vez
  const note = document.getElementById('face-note');
  if (note) note.remove();

  const cuentaPrefilled = document.getElementById('fm-cuenta').value;

  const titles = { estudiante:'Agregar alumno', maestro:'Agregar maestro', admin:'Agregar administrador' };
  document.getElementById('personModalTitle').textContent = titles[_editRole]||'Agregar';

  // Limpiar campos (menos el de cuenta que ya fue llenado)
  ['fm-nombre','fm-ap','fm-am','fm-correo','fm-grado','fm-grupo','fm-pwd'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  document.getElementById('fm-cuenta').value  = cuentaPrefilled;
  document.getElementById('fm-pwd').placeholder = 'Contraseña obligatoria';

  _configureForm(_editRole);
  _showModal('personModal');
}

// ── Configurar visibilidad de campos según rol ────────────────────────────────
function _configureForm(role) {
  document.getElementById('fm-correo-group').style.display =
    (role === 'maestro' || role === 'admin') ? '' : 'none';
  document.getElementById('fm-grado-grupo-group').style.display =
    role === 'estudiante' ? '' : 'none';
  document.getElementById('fm-pwd-group').style.display =
    role === 'estudiante' ? 'none' : '';
}


/* ══════════════════════════════════════════════════════
   GUARDAR (crear o editar)
══════════════════════════════════════════════════════ */

async function crudSave() {
  const nombre = document.getElementById('fm-nombre').value.trim();
  const ap     = document.getElementById('fm-ap').value.trim();
  const am     = document.getElementById('fm-am').value.trim();
  const cuenta = document.getElementById('fm-cuenta').value.trim();
  const correo = document.getElementById('fm-correo').value.trim();
  const grado  = document.getElementById('fm-grado').value.trim();
  const grupo  = document.getElementById('fm-grupo').value.trim();
  const pwd    = document.getElementById('fm-pwd').value;

  if (!nombre || !ap) { _toast('⚠ Nombre y apellido paterno son obligatorios.', 'warn'); return; }

  if (_editId === null && (_editRole === 'maestro' || _editRole === 'admin') && !pwd) {
    _toast('⚠ La contraseña es obligatoria al crear un ' + _editRole + '.', 'warn'); return;
  }

  const sesion = _getSession();
  const actPor = sesion ? sesion.id : null;
  const nowStr = _now();

  try {
    await _ensureDB();

    if (_editId === null) {
      const pwdHash = pwd ? await _sha256(pwd) : null;
      _run(`
        INSERT INTO usuarios
          (nombre, apellido_paterno, apellido_materno, numero_cuenta, correo,
           rol, contrasena, registrado_por, fecha_registro,
           actualizado_por, fecha_actualizacion)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
      `, [nombre, ap, am||'.', cuenta||null, correo||null,
          _editRole, pwdHash, actPor, nowStr, actPor, nowStr]);

      const newId = _lastId();
      if (_editRole === 'estudiante' && newId && grado) {
        _run('INSERT INTO estudiantes_detalle (usuario_id, grado, grupo) VALUES (?,?,?)',
          [newId, parseInt(grado,10), grupo||'']);
      }
      _toast(`✓ ${nombre} ${ap} registrado correctamente.`, 'ok');

    } else {
      const pwdHash = pwd ? await _sha256(pwd) : null;
      if (pwdHash) {
        _run(`UPDATE usuarios SET nombre=?,apellido_paterno=?,apellido_materno=?,
               numero_cuenta=?,correo=?,contrasena=?,actualizado_por=?,fecha_actualizacion=?
               WHERE id=?`,
          [nombre,ap,am||'.',cuenta||null,correo||null,pwdHash,actPor,nowStr,_editId]);
      } else {
        _run(`UPDATE usuarios SET nombre=?,apellido_paterno=?,apellido_materno=?,
               numero_cuenta=?,correo=?,actualizado_por=?,fecha_actualizacion=?
               WHERE id=?`,
          [nombre,ap,am||'.',cuenta||null,correo||null,actPor,nowStr,_editId]);
      }

      if (_editRole === 'estudiante' && grado) {
        const det = _query('SELECT id FROM estudiantes_detalle WHERE usuario_id=?',[_editId]);
        if (det.length) {
          _run('UPDATE estudiantes_detalle SET grado=?,grupo=? WHERE usuario_id=?',
            [parseInt(grado,10),grupo||'',_editId]);
        } else {
          _run('INSERT INTO estudiantes_detalle (usuario_id,grado,grupo) VALUES (?,?,?)',
            [_editId,parseInt(grado,10),grupo||'']);
        }
      }
      _toast(`✓ ${nombre} ${ap} actualizado correctamente.`, 'ok');
    }

    _closeModal('personModal');
    _showExportBanner();

    const esAdmin = sesion && sesion.rol === 'admin';
    crudRenderStudents();
    if (esAdmin) { crudRenderTeachers(); crudRenderAdmins(); }

  } catch(e) {
    _toast('✗ Error: ' + e.message, 'error');
    console.error(e);
  }
}

// ── Confirmar eliminación ─────────────────────────────────────────────────────
async function crudConfirmDelete() {
  if (!_delId) return;
  try {
    await _ensureDB();
    _run('DELETE FROM estudiantes_detalle WHERE usuario_id=?', [_delId]);
    _run('DELETE FROM usuarios WHERE id=?', [_delId]);
    _toast(`✓ ${_delName} eliminado.`, 'ok');
    _closeModal('deleteModal');
    _delId = null; _delName = '';
    _showExportBanner();
    const s = _getSession();
    crudRenderStudents();
    if (s && s.rol === 'admin') { crudRenderTeachers(); crudRenderAdmins(); }
  } catch(e) {
    _toast('✗ Error: ' + e.message, 'error');
  }
}


/* ══════════════════════════════════════════════════════
   CÁMARA — stop
══════════════════════════════════════════════════════ */

function _stopCamera() {
  clearInterval(_captureTimer);
  _captureTimer = null;
  if (_videoStream) {
    _videoStream.getTracks().forEach(t => t.stop());
    _videoStream = null;
  }
  const video = document.getElementById('face-video');
  if (video) video.srcObject = null;
}


/* ══════════════════════════════════════════════════════
   HELPERS DE MODALES Y UI
══════════════════════════════════════════════════════ */

function _showModal(id) { document.getElementById(id)?.classList.add('open'); }

function _closeModal(id) {
  document.getElementById(id)?.classList.remove('open');
  if (id === 'faceModal') _stopCamera();
}

function crudCloseModal(id) { _closeModal(id); }

function _toast(text, type) {
  let c = document.querySelector('.toast-container');
  if (!c) { c = document.createElement('div'); c.className = 'toast-container'; document.body.appendChild(c); }
  const id = 'ct-' + Date.now();
  const colors = { ok:'var(--accent)', warn:'var(--amber)', error:'var(--red)' };
  const el = document.createElement('div');
  el.className = 'toast'; el.id = id;
  el.style.borderLeftColor = colors[type]||'var(--accent)';
  el.innerHTML = `
    <div class="toast-body">
      <div class="toast-msg" style="color:var(--text-1)">${text}</div>
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>`;
  c.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

function _showExportBanner(msg) {
  let b = document.getElementById('export-banner');
  if (b) return;
  b = document.createElement('div');
  b.id = 'export-banner';
  b.className = 'export-banner';
  b.innerHTML = `
    <span>${msg || '⚠ Los cambios están en memoria. Descarga la BD para guardarlos permanentemente.'}</span>
    <button onclick="_exportDB()">💾 Descargar BD</button>
    <button onclick="this.parentElement.remove()" style="background:none;border:none;color:var(--text-3);cursor:pointer;padding:0 8px">✕</button>`;
  document.body.appendChild(b);
}
