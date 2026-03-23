/* ═══════════════════════════════════════════════════════
   data.js — Datos de ejemplo del laboratorio.
   Reemplaza con tu fuente real (API, BD).
═══════════════════════════════════════════════════════ */

/* ================================================================
   app.js — LabControl: Sistema de Gestión de Laboratorio
   Todo el JavaScript del proyecto en un solo archivo.

   ÍNDICE:
   1.  Datos del laboratorio (reemplazar con tu API/BD)
       1a. Personas dentro del lab ahora
       1b. Historial de accesos del día
       1c. Top visitantes del mes
       1d. Estadísticas por carrera
       1e. Estadísticas por hora
   2.  Funciones utilitarias (fechas, formatos)
   3.  Sistema de alertas de permanencia
       3a. Revisión y conteo de alertas
       3b. Renderizado de listas de alerta
       3c. Modal y toast de alerta
       3d. Configurar límite de tiempo
   4.  Renderizado del dashboard
       4a. KPIs
       4b. Lista de personas dentro (widget)
       4c. Tabla completa "dentro del lab"
       4d. Tabla registro de accesos (con filtros)
       4e. Tabla de accesos recientes
       4f. Estadísticas (top, barras, horas)
   5.  Navegación entre vistas
   6.  Sidebar (abrir/cerrar)
   7.  Modales (abrir/cerrar)
   8.  Panel de notificaciones (campana)
   9.  Reloj y fecha en tiempo real
   10. Modo oscuro / claro
   11. Actualización automática de datos
   12. Inicialización (DOMContentLoaded)
================================================================ */

'use strict';


/* ================================================================
   1. DATOS DEL LABORATORIO
   ⚠ IMPORTANTE: Estos son datos de ejemplo.
   En producción, reemplaza con llamadas a tu API:
     fetch('/api/lab/inside').then(r => r.json()).then(data => { ... })
================================================================ */

/* ── 1a. Personas actualmente dentro del laboratorio ── */
const INSIDE_NOW = [
  {
    id:        'U001',
    name:      'Fernanda Ruiz López',
    initials:  'FR',
    matricula: '220456789',
    career:    'Ing. en Sistemas',
    entryTime: getMinutesAgo(25),    // entró hace 25 min — sin alerta
  },
  {
    id:        'U002',
    name:      'Carlos Mendez Torres',
    initials:  'CM',
    matricula: '219877643',
    career:    'Ing. Industrial',
    entryTime: getMinutesAgo(142),   // lleva 2h 22min → DISPARA ALERTA
  },
  {
    id:        'U003',
    name:      'Ana Patricia Gómez',
    initials:  'AG',
    matricula: '221003412',
    career:    'Lic. Informática',
    entryTime: getMinutesAgo(55),    // 55 min — sin alerta
  },
  {
    id:        'U004',
    name:      'Rodrigo Sánchez Vela',
    initials:  'RS',
    matricula: '220981234',
    career:    'Ing. en Sistemas',
    entryTime: getMinutesAgo(130),   // 2h 10min → DISPARA ALERTA
  },
  {
    id:        'U005',
    name:      'Mariana Delgado Cruz',
    initials:  'MD',
    matricula: '222145670',
    career:    'Ing. Civil',
    entryTime: getMinutesAgo(10),    // 10 min — sin alerta
  }
];

/* ── 1b. Historial de accesos del día ── */
const ACCESS_LOG = [
  { id: 1,  name: 'Fernanda Ruiz López',   matricula: '220456789', career: 'Ing. en Sistemas', type: 'Entrada', time: '07:12', duration: null },
  { id: 2,  name: 'Carlos Mendez Torres',  matricula: '219877643', career: 'Ing. Industrial',  type: 'Entrada', time: '05:08', duration: null },
  { id: 3,  name: 'Paola Jiménez Reyes',   matricula: '218654321', career: 'Lic. Informática', type: 'Entrada', time: '07:00', duration: null },
  { id: 4,  name: 'Paola Jiménez Reyes',   matricula: '218654321', career: 'Lic. Informática', type: 'Salida',  time: '08:45', duration: '1h 45min' },
  { id: 5,  name: 'Luis Herrera Montes',   matricula: '220123456', career: 'Ing. en Sistemas', type: 'Entrada', time: '08:30', duration: null },
  { id: 6,  name: 'Luis Herrera Montes',   matricula: '220123456', career: 'Ing. en Sistemas', type: 'Salida',  time: '09:55', duration: '1h 25min' },
  { id: 7,  name: 'Ana Patricia Gómez',    matricula: '221003412', career: 'Lic. Informática', type: 'Entrada', time: '09:40', duration: null },
  { id: 8,  name: 'Diana Flores Ortega',   matricula: '221567890', career: 'Ing. Industrial',  type: 'Entrada', time: '10:05', duration: null },
  { id: 9,  name: 'Diana Flores Ortega',   matricula: '221567890', career: 'Ing. Industrial',  type: 'Salida',  time: '11:30', duration: '1h 25min' },
  { id: 10, name: 'Rodrigo Sánchez Vela',  matricula: '220981234', career: 'Ing. en Sistemas', type: 'Entrada', time: '05:20', duration: null },
  { id: 11, name: 'Mariana Delgado Cruz',  matricula: '222145670', career: 'Ing. Civil',        type: 'Entrada', time: '07:25', duration: null },
  { id: 12, name: 'Joel Ramírez Estrada',  matricula: '219456789', career: 'Ing. en Sistemas', type: 'Entrada', time: '11:00', duration: null },
  { id: 13, name: 'Joel Ramírez Estrada',  matricula: '219456789', career: 'Ing. en Sistemas', type: 'Salida',  time: '12:10', duration: '1h 10min' },
  { id: 14, name: 'Karla Vega Moreno',     matricula: '220876543', career: 'Lic. Informática', type: 'Entrada', time: '12:00', duration: null },
  { id: 15, name: 'Karla Vega Moreno',     matricula: '220876543', career: 'Lic. Informática', type: 'Salida',  time: '13:05', duration: '1h 05min' }
];

/* ── 1c. Top visitantes del mes ── */
const TOP_VISITORS = [
  { id: 'U002', name: 'Carlos Mendez Torres', initials: 'CM', career: 'Ing. Industrial',  visits: 28 },
  { id: 'U004', name: 'Rodrigo Sánchez Vela', initials: 'RS', career: 'Ing. en Sistemas', visits: 24 },
  { id: 'U003', name: 'Ana Patricia Gómez',   initials: 'AG', career: 'Lic. Informática', visits: 21 },
  { id: 'U001', name: 'Fernanda Ruiz López',  initials: 'FR', career: 'Ing. en Sistemas', visits: 19 },
  { id: 'U006', name: 'Paola Jiménez Reyes',  initials: 'PJ', career: 'Lic. Informática', visits: 17 },
  { id: 'U007', name: 'Luis Herrera Montes',  initials: 'LH', career: 'Ing. en Sistemas', visits: 15 },
  { id: 'U008', name: 'Diana Flores Ortega',  initials: 'DF', career: 'Ing. Industrial',  visits: 13 }
];

/* ── 1d. Accesos por carrera ── */
const CAREER_STATS = [
  { label: 'Ing. en Sistemas', count: 89, color: 'accent' },
  { label: 'Lic. Informática', count: 56, color: 'blue'   },
  { label: 'Ing. Industrial',  count: 41, color: 'amber'  },
  { label: 'Ing. Civil',       count: 22, color: 'purple' }
];

/* ── 1e. Accesos por hora del día ── */
const HOUR_STATS = [
  { hour: '06', count: 3  },
  { hour: '07', count: 18 },
  { hour: '08', count: 34 },
  { hour: '09', count: 42 },  // hora pico
  { hour: '10', count: 38 },
  { hour: '11', count: 29 },
  { hour: '12', count: 20 },
  { hour: '13', count: 15 },
  { hour: '14', count: 10 },
  { hour: '15', count: 6  }
];


/* ================================================================
   2. FUNCIONES UTILITARIAS
================================================================ */

/** Devuelve un Date que está N minutos en el pasado (para datos de prueba) */
function getMinutesAgo(n) {
  const d = new Date();
  d.setMinutes(d.getMinutes() - n);
  return d;
}

/** Calcula cuántos minutos han pasado desde un Date hasta ahora */
function minutesElapsed(dateObj) {
  return Math.floor((Date.now() - dateObj.getTime()) / 60000);
}

/** Convierte minutos a formato legible: 90 → "1h 30min", 45 → "45 min" */
function formatDuration(mins) {
  if (mins < 60) return `${mins} min`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return m > 0 ? `${h}h ${m}min` : `${h}h`;
}

/** Formatea un Date a "HH:MM" en 24h */
function formatTime(dateObj) {
  return dateObj.toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit', hour12: false });
}

/** Convierte string de duración a minutos: "1h 25min" → 85 */
function parseDuration(str) {
  if (!str) return 0;
  let mins = 0;
  const hMatch = str.match(/(\d+)h/);
  const mMatch = str.match(/(\d+)\s*min/);
  if (hMatch) mins += parseInt(hMatch[1], 10) * 60;
  if (mMatch) mins += parseInt(mMatch[1], 10);
  return mins;
}

/** Inserta texto en un elemento por ID (helper general) */
function setEl(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}