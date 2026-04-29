"""
api.py — Puente Flask entre el Dashboard web y el sistema facial Python.
========================================================================
Expone endpoints HTTP que el navegador llama para:
  1. Verificar si una cuenta ya tiene vectores registrados.
  2. Reconocer un rostro contra los vectores de una cuenta específica.

INSTALACIÓN:
  pip install flask flask-cors

USO:
  Coloca este archivo en la raíz del proyecto (junto a face_engine.py
  y database.py), luego ejecuta:
    python api.py
  El servidor corre en http://localhost:5050
  (Puerto 5050 para no chocar con otros servicios).

ESTRUCTURA ESPERADA DEL PROYECTO:
  proyecto/
    api.py              ← este archivo
    face_engine.py
    database.py
    models/             ← w600k_mbf.onnx, yunet, etc.
    database/
      reconocimiento_facial.db
    Dashboard/          ← el frontend web
"""

import os
import sys
import base64
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Rutas ────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)   # asegurar que se encuentren face_engine y database

from face_engine import extraer_caracteristicas, distancia_coseno, VECTOR_DIM
from database   import (
    conectar,
    cargar_vectores_por_angulo,
    reconocer_persona,
    UMBRAL,
    RECHAZO,
    GAP_MIN,
)

# ── App Flask ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")   # permite llamadas desde cualquier origen local


# =============================================================================
#  UTILIDADES
# =============================================================================

def _imagen_de_b64(b64_str: str) -> np.ndarray | None:
    """
    Convierte un string base64 (data:image/...;base64,<datos> o solo <datos>)
    a un array BGR de OpenCV.
    """
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        data = base64.b64decode(b64_str)
        arr  = np.frombuffer(data, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[API] Error decodificando imagen: {e}")
        return None


def _vectores_de_cuenta(numero_cuenta: str) -> list[dict]:
    """
    Devuelve todos los registros de vectores_por_angulo para el
    usuario con ese numero_cuenta (puede ser más de uno por ángulo).
    """
    conn = conectar()
    cur  = conn.cursor()
    cur.execute("""
        SELECT va.angulo, va.vector, va.n_muestras, va.dimensiones,
               u.id, u.nombre, u.apellido_paterno, u.rol
        FROM vectores_por_angulo va
        JOIN usuarios u ON u.id = va.usuario_id
        WHERE u.numero_cuenta = ?
    """, (numero_cuenta.strip(),))
    rows = cur.fetchall()
    conn.close()

    resultado = []
    for angulo, vjson, n, dims, uid, nombre, ap, rol in rows:
        if dims != VECTOR_DIM:
            continue
        resultado.append({
            "angulo":   angulo,
            "vector":   np.array(json.loads(vjson), dtype=np.float32),
            "n":        n,
            "id":       uid,
            "nombre":   f"{nombre} {ap}",
            "rol":      rol,
        })
    return resultado


# =============================================================================
#  ENDPOINTS
# =============================================================================

@app.route("/api/ping", methods=["GET"])
def ping():
    """Verificar que la API está corriendo."""
    return jsonify({"ok": True, "msg": "API facial activa"})


# ---------------------------------------------------------------------------
@app.route("/api/tiene_vectores/<numero_cuenta>", methods=["GET"])
def tiene_vectores(numero_cuenta: str):
    """
    GET /api/tiene_vectores/12345678

    Devuelve si la cuenta tiene vectores registrados y el nombre del usuario.
    El Dashboard lo llama al escribir el número de cuenta para saber si
    habrá verificación facial o no.
    """
    registros = _vectores_de_cuenta(numero_cuenta)
    if not registros:
        return jsonify({
            "tiene_vectores": False,
            "nombre":         None,
            "angulos":        [],
        })

    angulos = list({r["angulo"] for r in registros})
    return jsonify({
        "tiene_vectores": True,
        "nombre":         registros[0]["nombre"],
        "angulos":        sorted(angulos),
    })


# ---------------------------------------------------------------------------
@app.route("/api/reconocer_cuenta", methods=["POST"])
def reconocer_cuenta():
    """
    POST /api/reconocer_cuenta
    Body JSON:
      {
        "imagen":        "<base64 del frame>",
        "numero_cuenta": "12345678"
      }

    Proceso:
      1. Decodifica la imagen.
      2. Extrae el embedding con MobileFaceNet (el mismo modelo que usa
         interfaz.py / face_engine.py).
      3. Compara con TODOS los vectores almacenados para esa cuenta.
      4. Devuelve si coincide o no.
    """
    body = request.get_json(force=True, silent=True) or {}

    imagen_b64   = body.get("imagen", "")
    numero_cuenta = body.get("numero_cuenta", "").strip()

    if not imagen_b64:
        return jsonify({"ok": False, "error": "Se requiere el campo 'imagen'"}), 400
    if not numero_cuenta:
        return jsonify({"ok": False, "error": "Se requiere 'numero_cuenta'"}), 400

    # 1. Decodificar imagen
    frame = _imagen_de_b64(imagen_b64)
    if frame is None:
        return jsonify({"ok": False, "error": "No se pudo decodificar la imagen"}), 400

    # 2. Extraer embedding con face_engine (MobileFaceNet)
    embedding, bbox, tipo = extraer_caracteristicas(frame)

    if embedding is None:
        return jsonify({
            "ok":            False,
            "coincide":      False,
            "error":         "No se detectó ningún rostro en la imagen.",
            "tiene_rostro":  False,
        })

    # 3. Cargar vectores de la cuenta
    registros = _vectores_de_cuenta(numero_cuenta)
    if not registros:
        # La cuenta no tiene vectores → no hay contra qué comparar
        return jsonify({
            "ok":            True,
            "coincide":      None,    # None = sin datos, no es fallo
            "sin_vectores":  True,
            "nombre":        None,
            "distancia":     None,
            "similitud_pct": None,
            "msg": (
                "Esta persona aún no tiene rostro registrado en el sistema. "
                "Pídele que se registre primero en el terminal biométrico."
            ),
        })

    # 4. Comparar contra todos los ángulos disponibles
    mejor_dist   = float("inf")
    mejor_angulo = None
    for reg in registros:
        dist = distancia_coseno(embedding, reg["vector"])
        if dist < mejor_dist:
            mejor_dist   = dist
            mejor_angulo = reg["angulo"]

    nombre_persona = registros[0]["nombre"]
    sim_raw  = max(0.0, 1.0 - (mejor_dist / 2.0))   # 0-1
    sim_pct  = round(sim_raw * 100, 1)
    coincide = mejor_dist <= UMBRAL

    print(
        f"[API] cuenta={numero_cuenta} "
        f"dist={mejor_dist:.4f} sim={sim_pct}% "
        f"angulo={mejor_angulo} coincide={coincide}"
    )

    return jsonify({
        "ok":            True,
        "coincide":      coincide,
        "sin_vectores":  False,
        "nombre":        nombre_persona,
        "distancia":     round(float(mejor_dist), 4),
        "similitud_pct": sim_pct,
        "angulo_match":  mejor_angulo,
        "umbral":        UMBRAL,
        "msg": (
            f"Coincide con {nombre_persona} ({sim_pct}%)"
            if coincide else
            f"No coincide. Mejor candidato: {nombre_persona} ({sim_pct}%)"
        ),
    })


# ---------------------------------------------------------------------------
@app.route("/api/reconocer_libre", methods=["POST"])
def reconocer_libre():
    """
    POST /api/reconocer_libre
    Body JSON: { "imagen": "<base64>" }

    Reconocimiento sin filtrar por cuenta — contra TODA la base.
    Útil para el modo ACCESO del dashboard.
    """
    body       = request.get_json(force=True, silent=True) or {}
    imagen_b64 = body.get("imagen", "")

    if not imagen_b64:
        return jsonify({"ok": False, "error": "Falta 'imagen'"}), 400

    frame = _imagen_de_b64(imagen_b64)
    if frame is None:
        return jsonify({"ok": False, "error": "Imagen inválida"}), 400

    embedding, bbox, tipo = extraer_caracteristicas(frame)
    if embedding is None:
        return jsonify({"ok": False, "tiene_rostro": False,
                        "error": "Sin rostro detectado"})

    resultado = reconocer_persona(embedding, angulo_nuevo=(tipo or "frontal"))

    if resultado is None:
        return jsonify({"ok": True, "reconocido": False,
                        "msg": "Persona no reconocida"})

    return jsonify({
        "ok":          True,
        "reconocido":  True,
        "acceso":      resultado["acceso"],
        "nombre":      resultado["nombre"],
        "cuenta":      resultado["numero_cuenta"],
        "rol":         resultado["rol"],
        "similitud":   resultado["similitud_pct"],
        "distancia":   resultado["distancia"],
    })


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 56)
    print("  API Facial — LabControl")
    print("  http://localhost:5050")
    print("  Ctrl+C para detener")
    print("=" * 56)
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
