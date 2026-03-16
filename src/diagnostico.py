"""
diagnostico.py
==============
Muestra en tiempo real el angulo que detecta el sistema.
No requiere MediaPipe.

Teclas:
  Q     — salir
  Y/U   — bajar/subir umbral YAW (giro lateral)
  I/O   — bajar/subir umbral ABAJO (inclinacion)
  R     — reset umbrales al valor original
  D     — medir distancias contra BD (5 muestras)
"""
import cv2
import numpy as np
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── camara ───────────────────────────────────────────────────────────────────
USAR_PICAM = False
try:
    from picamera2 import Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"}))
    picam2.start()
    time.sleep(0.5)
    USAR_PICAM = True
    print("[CAM] Picamera2 detectada")
except Exception:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[CAM] Webcam OpenCV")

def leer_frame():
    if USAR_PICAM:
        return picam2.capture_array()
    ret, f = cap.read()
    return f if ret else None

# ─── importar motor facial ────────────────────────────────────────────────────
from face_engine import (extraer_caracteristicas, distancia_ponderada,
                          _detectar_dnn, _clasificar_por_asimetria,
                          preprocesar_cara,
                          TIPO_FRONTAL, TIPO_PERFIL_D, TIPO_PERFIL_I, TIPO_ABAJO)

# ─── umbrales ajustables ──────────────────────────────────────────────────────
UMBRAL_YAW   = 0.10
UMBRAL_ABAJO = 0.12

COLORES = {
    TIPO_FRONTAL:  (0,   212, 255),
    TIPO_PERFIL_D: (0,   165, 255),
    TIPO_PERFIL_I: (255, 165,   0),
    TIPO_ABAJO:    (180,   0, 255),
    "sin_cara":    (80,   80,  80),
}

# ─── BD ───────────────────────────────────────────────────────────────────────
try:
    from database import cargar_vectores_por_angulo
    registros_bd = cargar_vectores_por_angulo()
    print(f"[BD] {len(registros_bd)} vectores")
    for r in registros_bd:
        print(f"     {r['nombre']} — {r['angulo']}")
except Exception as e:
    registros_bd = []
    print(f"[BD] {e}")

muestras_dist   = []
modo_distancias = False

print("\nTeclas: Q=salir  Y/U=yaw  I/O=abajo  R=reset  D=distancias\n")

# ─── loop ─────────────────────────────────────────────────────────────────────
while True:
    frame = leer_frame()
    if frame is None:
        continue
    frame = cv2.flip(frame, 1)
    h_img, w_img = frame.shape[:2]

    caras = _detectar_dnn(frame)

    tipo    = "sin_cara"
    ratio_h = 0.0
    ratio_v = 0.0
    bbox    = None

    if caras:
        x, y, w, h, conf = caras[0]
        bbox = (x, y, w, h)

        gris    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        recorte = preprocesar_cara(gris[y:y+h, x:x+w])
        if recorte.size > 0:
            cara128 = cv2.resize(recorte, (128, 128))

            gx = cv2.Sobel(cara128, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(cara128, cv2.CV_32F, 0, 1, ksize=3)
            bordes = np.abs(gx) + np.abs(gy)
            hw, ww = bordes.shape
            margen = ww // 10
            izq = float(np.mean(bordes[:, :ww//2 - margen]))
            der = float(np.mean(bordes[:, ww//2 + margen:]))
            ratio_h = (izq - der) / (izq + der + 1e-6)

            t1 = float(np.mean(bordes[:hw//3,   :]))
            t3 = float(np.mean(bordes[2*hw//3:, :]))
            ratio_v = (t1 - t3) / (t1 + t3 + 1e-6)

            if ratio_h > UMBRAL_YAW:
                tipo = TIPO_PERFIL_D
            elif ratio_h < -UMBRAL_YAW:
                tipo = TIPO_PERFIL_I
            elif ratio_v > UMBRAL_ABAJO:
                tipo = TIPO_ABAJO
            elif ratio_v < -UMBRAL_ABAJO:
                tipo = TIPO_ABAJO
            else:
                tipo = TIPO_FRONTAL

            if modo_distancias and registros_bd:
                v, p, coords, t = extraer_caracteristicas(frame)
                if v is not None:
                    muestras_dist.append((v, p))
                    if len(muestras_dist) >= 5:
                        print("\n--- DISTANCIAS ---")
                        vf = np.mean([m[0] for m in muestras_dist], axis=0)
                        pf = np.mean([m[1] for m in muestras_dist], axis=0)
                        por_persona = {}
                        for reg in registros_bd:
                            pid = reg["persona_id"]
                            dist, nz = distancia_ponderada(vf, pf,
                                           reg["vector"], reg["pesos"])
                            if pid not in por_persona or dist < por_persona[pid][0]:
                                por_persona[pid] = (dist, nz,
                                    reg["nombre"], reg["angulo"])
                        for pid, (dist, nz, nombre, angulo) in por_persona.items():
                            estado = ("ACCESO" if dist <= 0.12 else
                                      "CERCA"  if dist <= 0.25 else "LEJOS")
                            print(f"  {nombre:20s} dist={dist:.4f} "
                                  f"angulo={angulo:12s} [{estado}]")
                        print()
                        muestras_dist   = []
                        modo_distancias = False

    # ── Dibujar ───────────────────────────────────────────────────────────────
    color = COLORES.get(tipo, (80, 80, 80))

    if bbox:
        x, y, w, h = bbox
        L = max(15, w // 4)
        for p1, p2 in [
            ((x,   y),   (x+L, y)),    ((x,   y),   (x,   y+L)),
            ((x+w, y),   (x+w-L, y)),  ((x+w, y),   (x+w, y+L)),
            ((x,   y+h), (x+L, y+h)),  ((x,   y+h), (x,   y+h-L)),
            ((x+w, y+h), (x+w-L,y+h)), ((x+w, y+h), (x+w, y+h-L)),
        ]:
            cv2.line(frame, p1, p2, color, 2)
        cv2.putText(frame, tipo.upper(), (x, y+h+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def txt(s, yy, col=(200, 200, 200)):
        cv2.putText(frame, s, (10, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

    txt(f"ANGULO: {tipo.upper()}", 22, color)
    txt(f"ratio_h (yaw):   {ratio_h:+.3f}  umbral +/-{UMBRAL_YAW:.2f}  [Y/U]", 44)
    txt(f"ratio_v (pitch): {ratio_v:+.3f}  umbral +/-{UMBRAL_ABAJO:.2f}  [I/O]", 64)
    txt("ratio_h>+u=PERFIL_DER  <-u=PERFIL_IZQ", 84, (140, 140, 140))
    txt("ratio_v>+u=ABAJO       <-u=ABAJO",       100, (140, 140, 140))

    if modo_distancias:
        txt(f"Midiendo... {len(muestras_dist)}/5", 120, (0, 255, 136))

    cv2.imshow("Diagnostico — Q salir", frame)
    key = cv2.waitKey(1) & 0xFF

    if   key == ord('q'): break
    elif key == ord('y'):
        UMBRAL_YAW = max(0.02, UMBRAL_YAW - 0.01)
        print(f"[YAW] {UMBRAL_YAW:.3f}")
    elif key == ord('u'):
        UMBRAL_YAW = min(0.40, UMBRAL_YAW + 0.01)
        print(f"[YAW] {UMBRAL_YAW:.3f}")
    elif key == ord('i'):
        UMBRAL_ABAJO = max(0.05, UMBRAL_ABAJO - 0.01)
        print(f"[ABAJO] {UMBRAL_ABAJO:.3f}")
    elif key == ord('o'):
        UMBRAL_ABAJO = min(0.50, UMBRAL_ABAJO + 0.01)
        print(f"[ABAJO] {UMBRAL_ABAJO:.3f}")
    elif key == ord('r'):
        UMBRAL_YAW, UMBRAL_ABAJO = 0.10, 0.12
        print("[RESET]")
    elif key == ord('d'):
        modo_distancias, muestras_dist = True, []
        print("[DIST] Midiendo 5 muestras...")

# ─── cierre ───────────────────────────────────────────────────────────────────
if USAR_PICAM:
    picam2.close()
else:
    cap.release()
cv2.destroyAllWindows()

print(f"\nUmbrales finales — copia a face_engine.py si funcionan bien:")
print(f"  UMBRAL_YAW   = {UMBRAL_YAW:.3f}")
print(f"  UMBRAL_ABAJO = {UMBRAL_ABAJO:.3f}")