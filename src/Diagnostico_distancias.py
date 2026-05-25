"""
diagnostico_distancias.py
=========================
Muestra en tiempo real las distancias chi-cuadrado entre la cara
detectada y cada persona registrada en la base de datos.

Corre: python3 diagnostico_distancias.py
Ctrl+C para salir

USO PARA CALIBRAR UMBRAL y UMBRAL_RECHAZO:
  1. Pon tu cara (persona registrada) → anota la distancia minima
  2. Pon otra cara (NO registrada)    → anota la distancia minima
  3. En database.py ajusta:
       UMBRAL        = valor entre los dos grupos (acceso si <= umbral)
       UMBRAL_RECHAZO = un poco por encima de la distancia de persona NO registrada

Ejemplo:
  Persona registrada   → dist minima ~0.6
  Persona NO registrada → dist minima ~1.8
  → UMBRAL = 1.0  (acepta registrados, rechaza no registrados)
  → UMBRAL_RECHAZO = 1.5 (si dist > 1.5, retorna None directamente)
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

from face_engine import extraer_caracteristicas
from database import (
    distancia_chi2,
    cargar_vectores_por_angulo,
    UMBRAL,
    RECHAZO as UMBRAL_RECHAZO,   # alias para no cambiar el resto del archivo
)

print("\n" + "="*70)
print("  DIAGNOSTICO DISTANCIAS — calibracion de UMBRAL y UMBRAL_RECHAZO")
print("="*70)
print(f"  UMBRAL actual        = {UMBRAL}  (acceso si distancia <= este valor)")
print(f"  UMBRAL_RECHAZO actual = {UMBRAL_RECHAZO}  (desconocido si distancia > este valor)")
print("="*70)
print()

registros = cargar_vectores_por_angulo()
if not registros:
    print("  ERROR: No hay usuarios registrados en la base de datos.")
    print("  Registra al menos un usuario primero.")
    picam2.close()
    sys.exit(1)

# Agrupar por usuario para mostrar el mejor angulo por persona
usuarios = {}
for reg in registros:
    uid = reg["usuario_id"]
    if uid not in usuarios:
        usuarios[uid] = {"nombre": reg["nombre"], "vectores": []}
    usuarios[uid]["vectores"].append(reg["vector"])

print(f"  Usuarios en BD: {len(usuarios)}")
for uid, info in usuarios.items():
    print(f"    ID={uid}  {info['nombre']}  ({len(info['vectores'])} angulos)")
print()
print(f"  {'DIST_MIN':>8}  {'MEJOR_MATCH':<22}  {'DECISION':<20}  detalle")
print("  " + "-"*72)

n = 0
try:
    while True:
        frame = cv2.flip(picam2.capture_array(), 1)
        n += 1
        if n % 6 != 0:   # cada ~0.3s para no saturar la terminal
            time.sleep(0.05)
            continue

        vector, bbox, tipo = extraer_caracteristicas(frame)

        if vector is None:
            print(f"  {'---':>8}  {'SIN CARA':<22}  {'---':<20}")
            time.sleep(0.15)
            continue

        # Calcular distancia minima a cada usuario
        mejores = {}
        for reg in registros:
            uid  = reg["usuario_id"]
            dist = distancia_chi2(vector, reg["vector"])
            if uid not in mejores or dist < mejores[uid]["dist"]:
                mejores[uid] = {
                    "nombre": reg["nombre"],
                    "dist":   dist,
                    "angulo": reg["angulo"],
                }

        if not mejores:
            continue

        # El mas cercano
        mejor = min(mejores.values(), key=lambda x: x["dist"])
        dist_min = mejor["dist"]
        nombre   = mejor["nombre"][:22]

        # Decision segun umbrales actuales
        if dist_min <= UMBRAL:
            decision = "✓ ACCESO PERMITIDO"
        elif dist_min <= UMBRAL_RECHAZO:
            decision = "✗ DENEGADO"
        else:
            decision = "? DESCONOCIDO (None)"

        # Mostrar todas las distancias en una línea compacta
        todas = "  ".join(
            f"{v['nombre'].split()[0]}:{v['dist']:.3f}"
            for v in sorted(mejores.values(), key=lambda x: x["dist"])
        )

        print(f"  {dist_min:>8.4f}  {nombre:<22}  {decision:<20}  [{todas}]")

except KeyboardInterrupt:
    print("\n\n[Ctrl+C] Saliendo...")
    print()
    print("  Resumen para calibrar:")
    print(f"    UMBRAL actual        = {UMBRAL}")
    print(f"    UMBRAL_RECHAZO actual = {UMBRAL_RECHAZO}")
    print()
    print("  Ajusta en database.py:")
    print("    UMBRAL        → un poco por encima de la dist de tu cara registrada")
    print("    UMBRAL_RECHAZO → un poco por debajo de la dist de caras NO registradas")
finally:
    picam2.close()