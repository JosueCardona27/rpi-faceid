// ── Config ─────────────────────────────────────────────────────────────────
// Ruta al archivo .db relativa a desde donde se sirve el HTML.
// Si login.html esta en /web/ y el proyecto en la raiz, la ruta es:
const DB_PATH      = "../database/reconocimiento_facial.db";
const DASHBOARD    = "dashboard.html";  // donde redirigir tras login exitoso

// ── Estado ─────────────────────────────────────────────────────────────────
let SQL = null;    // instancia de sql.js
let db  = null;    // base de datos cargada

const dbDot   = document.getElementById("dbDot");
const dbLabel = document.getElementById("dbLabel");
const msg     = document.getElementById("msg");
const btnLogin= document.getElementById("btnLogin");

// ── Inicializar sql.js y cargar la BD ──────────────────────────────────────
async function inicializar() {
  try {
    // 1. Inicializar el motor WebAssembly de sql.js
    SQL = await initSqlJs({
      locateFile: file =>
        `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.3/${file}`
    });

    // 2. Descargar el archivo .db via fetch
    const resp = await fetch(DB_PATH + "?t=" + Date.now()); // evitar cache
    if (!resp.ok) {
      throw new Error(`No se pudo cargar la BD (HTTP ${resp.status})`);
    }

    const buffer = await resp.arrayBuffer();
    db = new SQL.Database(new Uint8Array(buffer));

    // 3. Verificar que la tabla usuarios existe
    const test = db.exec(
      "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='usuarios'"
    );
    const count = test[0]?.values[0]?.[0] ?? 0;

    if (count === 0) {
      throw new Error("La BD no tiene la tabla 'usuarios'. Ejecuta esquema.sql.");
    }

    dbDot.className   = "db-dot ok";
    dbLabel.textContent = "Base de datos conectada";
    btnLogin.disabled = false;

  } catch (e) {
    dbDot.className   = "db-dot error";
    dbLabel.textContent = "Error: " + e.message;
    mostrarMsg("error",
      "No se pudo cargar la base de datos.<br>" +
      "Asegurate de servir los archivos con:<br>" +
      "<code>python -m http.server 8080</code><br>" +
      "desde la raiz del proyecto.");
    btnLogin.disabled = true;
  }
}

// ── Hash SHA-256 (mismo algoritmo que hashlib.sha256 en Python) ────────────
async function sha256(texto) {
  const encoded = new TextEncoder().encode(texto);
  const hashBuffer = await crypto.subtle.digest("SHA-256", encoded);
  const hashArray  = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, "0")).join("");
}

// ── Iniciar sesion ──────────────────────────────────────────────────────────
async function iniciarSesion() {
  const correo = document.getElementById("correo").value.trim().toLowerCase();
  const pwd    = document.getElementById("pwd").value;

  if (!correo || !pwd) {
    mostrarMsg("error", "Completa todos los campos.");
    return;
  }

  if (!db) {
    mostrarMsg("error", "La base de datos no esta cargada.");
    return;
  }

  btnLogin.classList.add("loading");
  btnLogin.disabled = true;
  ocultarMsg();

  try {
    // Calcular hash SHA-256 de la contrasena ingresada
    const hashPwd = await sha256(pwd);

    // Consultar usuario en la BD
    const result = db.exec(`
      SELECT id, nombre, apellido_paterno, apellido_materno,
             numero_cuenta, correo, rol
      FROM usuarios
      WHERE LOWER(correo) = '${correo.replace(/'/g, "''")}'
        AND contrasena = '${hashPwd}'
        AND rol IN ('admin', 'maestro')
      LIMIT 1
    `);

    if (!result || result.length === 0 || result[0].values.length === 0) {
      // Verificar si el correo existe pero la contrasena es incorrecta
      const chkCorreo = db.exec(`
        SELECT COUNT(*) FROM usuarios
        WHERE LOWER(correo) = '${correo.replace(/'/g, "''")}'
      `);
      const existeCorreo = chkCorreo[0]?.values[0]?.[0] > 0;

      mostrarMsg("error",
        existeCorreo
          ? "Contrasena incorrecta."
          : "Correo no encontrado o sin permisos de acceso.");
      return;
    }

    // Login exitoso — guardar datos en sessionStorage
    const cols = result[0].columns;
    const vals = result[0].values[0];
    const usuario = {};
    cols.forEach((col, i) => usuario[col] = vals[i]);

    sessionStorage.setItem("usuario_login", JSON.stringify(usuario));

    mostrarMsg("success",
      `Bienvenido, ${usuario.nombre} ${usuario.apellido_paterno} ` +
      `(${usuario.rol.toUpperCase()}).<br>Redirigiendo...`);

    setTimeout(() => {
      window.location.href = DASHBOARD;
    }, 1200);

  } catch (e) {
    mostrarMsg("error", "Error al consultar la BD: " + e.message);
  } finally {
    btnLogin.classList.remove("loading");
    btnLogin.disabled = false;
  }
}

// ── Helpers UI ──────────────────────────────────────────────────────────────
function mostrarMsg(tipo, texto) {
  msg.className = "msg " + tipo;
  msg.innerHTML = texto;
}

function ocultarMsg() {
  msg.className = "msg";
  msg.textContent = "";
}

// ── Enter en los campos dispara el login ────────────────────────────────────
document.getElementById("correo").addEventListener("keydown", e => {
  if (e.key === "Enter") document.getElementById("pwd").focus();
});
document.getElementById("pwd").addEventListener("keydown", e => {
  if (e.key === "Enter") iniciarSesion();
});

// ── Inicio ──────────────────────────────────────────────────────────────────
btnLogin.disabled = true;  // deshabilitado hasta que la BD cargue
inicializar();