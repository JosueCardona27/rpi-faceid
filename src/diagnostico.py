"""
diagnostico.py - muestra angulo REAL + score YuNet + varianza Laplaciano
=========================================================================
Corre: python3 diagnostico.py
Ctrl+C para salir

Columnas mostradas:
  TIPO      — angulo clasificado (frontal / perfil_der / perfil_izq)
  SCORE     — confianza de YuNet (0.0-1.0). Falsos positivos suelen < 0.45
  LAPLACIAN — varianza del Laplaciano del recorte. Cara real > fondo liso:
                fondo/pared IR  → tipicamente  3 - 20
                cara humana IR  → tipicamente 25 - 150
  UMBRAL_S  — _SCORE_MINIMO_CARA actual en face_engine
  UMBRAL_L  — _LAPLACIAN_MIN actual en face_engine

Uso para calibrar:
  1. Apunta la camara al fondo (sin nadie) → anota SCORE y LAPLACIAN
  2. Pon tu cara frente a la camara → anota SCORE y LAPLACIAN
  3. Ajusta _SCORE_MINIMO_CARA y _LAPLACIAN_MIN en face_engine.py
     a valores que queden entre los dos grupos de lecturas.
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "BGR888"}))
picam2.start()
time.sleep(0.5)

from face_engine import (
    _detectar_caras,
    _clasificar_angulo,
    _extraer_angulos_lbf,
    _varianza_laplaciano,
    _SCORE_MINIMO_CARA,
    _LAPLACIAN_MIN,
    _ultimo_face_yunet,
    TIPO_FRONTAL,
    TIPO_PERFIL_D,
    TIPO_PERFIL_I,
)
import face_engine as _fe

print("\n" + "="*72)
print("  DIAGNOSTICO — angulo + score YuNet + varianza Laplaciano")
print("="*72)
print(f"  Umbrales activos:  score >= {_SCORE_MINIMO_CARA}   laplacian >= {_LAPLACIAN_MIN}")
print("="*72)
print(f"  {'TIPO':<12} {'SCORE':>7} {'LAPLA':>7} {'UMBRAL_S':>9} {'UMBRAL_L':>9}  nota")
print("  " + "-"*60)

# Acceder al score de la ultima deteccion YuNet cruda (antes de filtros)
# Para eso usamos _yunet directamente para ver el score raw
from face_engine import _yunet, _init_detectores
_init_detectores()

n = 0
try:
    while True:
        frame = cv2.flip(picam2.capture_array(), 1)
        n += 1
        if n % 3 != 0:
            time.sleep(0.03)
            continue

        # ── Deteccion raw (sin filtros) para ver score y laplaciano reales ──
        h_img, w_img = frame.shape[:2]
        raw_detecciones = []

        if _yunet is not None:
            _yunet.setInputSize((w_img, h_img))
            _, faces = _yunet.detect(frame)
            if faces is not None and len(faces) > 0:
                for face in faces:
                    rx = max(0, int(face[0])); ry = max(0, int(face[1]))
                    rw = min(int(face[2]), w_img - rx)
                    rh = min(int(face[3]), h_img - ry)
                    rs = float(face[14])
                    if rw >= 15 and rh >= 15:
                        lap = _varianza_laplaciano(frame, rx, ry, rw, rh)
                        raw_detecciones.append((rx, ry, rw, rh, rs, lap, face))
                raw_detecciones.sort(key=lambda d: d[2]*d[3], reverse=True)

        if not raw_detecciones:
            # Tampoco detecto nada YuNet raw → SIN CARA real
            print(f"  {'SIN CARA':<12} {'---':>7} {'---':>7} "
                  f"{_SCORE_MINIMO_CARA:>9.2f} {_LAPLACIAN_MIN:>9.1f}")
            time.sleep(0.15)
            continue

        # Mostrar la deteccion mas grande (aunque no pase los filtros)
        rx, ry, rw, rh, rs, lap, face = raw_detecciones[0]

        pasa_score = rs   >= _SCORE_MINIMO_CARA
        pasa_lap   = lap  >= _LAPLACIAN_MIN
        pasa_ambos = pasa_score and pasa_lap

        # Clasificar angulo con esa deteccion
        _fe._ultimo_face_yunet = face
        bbox = (rx, ry, rw, rh)
        tipo = _clasificar_angulo(frame, bbox, frame.shape)

        if pasa_ambos:
            nota = "✓ ACEPTA"
        elif not pasa_score and not pasa_lap:
            nota = "✗ score+lap bajos"
        elif not pasa_score:
            nota = f"✗ score bajo  (necesita >={_SCORE_MINIMO_CARA:.2f})"
        else:
            nota = f"✗ lap baja    (necesita >={_LAPLACIAN_MIN:.1f})"

        print(f"  {tipo:<12} {rs:>7.3f} {lap:>7.1f} "
              f"{_SCORE_MINIMO_CARA:>9.2f} {_LAPLACIAN_MIN:>9.1f}  {nota}")

        time.sleep(0.15)

except KeyboardInterrupt:
    print("\n[Ctrl+C] Saliendo...")
finally:
    picam2.close()