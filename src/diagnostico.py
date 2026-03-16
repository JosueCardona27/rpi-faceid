"""
diagnostico.py
==============
Muestra en la TERMINAL los valores de angulo en tiempo real.
No usa ventana grafica — funciona en Raspberry Pi sin display.

Corre:  python3 diagnostico.py
Sal:    Ctrl+C
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
    print("[CAM] Picamera2 OK")
except Exception as e:
    print(f"[CAM] Picamera2 fallo ({e}), usando webcam")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def leer_frame():
    if USAR_PICAM:
        return picam2.capture_array()
    ret, f = cap.read()
    return f if ret else None

# ─── importar motor ───────────────────────────────────────────────────────────
from face_engine import (_detectar_dnn, preprocesar_cara,
                          TIPO_FRONTAL, TIPO_PERFIL_D,
                          TIPO_PERFIL_I, TIPO_ABAJO)

UMBRAL_YAW   = 0.10
UMBRAL_ABAJO = 0.12

print("\n" + "="*55)
print("  DIAGNOSTICO DE ANGULOS — valores en tiempo real")
print("="*55)
print("  Mueve la cabeza y observa los valores.")
print("  Ctrl+C para salir.\n")
print(f"  {'ANGULO':<14} {'ratio_h':>10} {'ratio_v':>10}  {'cara':>6}")
print("  " + "-"*50)

ultimo_tipo = None
n_frame     = 0

try:
    while True:
        frame = leer_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        frame   = cv2.flip(frame, 1)
        n_frame += 1

        # Solo analizar 1 de cada 3 frames para no saturar la terminal
        if n_frame % 3 != 0:
            time.sleep(0.03)
            continue

        caras = _detectar_dnn(frame)

        if not caras:
            if ultimo_tipo != "sin_cara":
                print(f"  {'SIN CARA':<14} {'---':>10} {'---':>10}  {'NO':>6}")
                ultimo_tipo = "sin_cara"
            time.sleep(0.1)
            continue

        x, y, w, h, conf = caras[0]
        gris    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        recorte = preprocesar_cara(gris[y:y+h, x:x+w])
        if recorte.size == 0:
            continue

        cara128 = cv2.resize(recorte, (128, 128))

        # Calcular ratios de asimetria
        gx = cv2.Sobel(cara128, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(cara128, cv2.CV_32F, 0, 1, ksize=3)
        bordes = np.abs(gx) + np.abs(gy)
        hw, ww = bordes.shape
        margen  = ww // 10
        izq     = float(np.mean(bordes[:, :ww//2 - margen]))
        der     = float(np.mean(bordes[:, ww//2 + margen:]))
        ratio_h = (izq - der) / (izq + der + 1e-6)

        t1      = float(np.mean(bordes[:hw//3,   :]))
        t3      = float(np.mean(bordes[2*hw//3:, :]))
        ratio_v = (t1 - t3) / (t1 + t3 + 1e-6)

        # Clasificar
        if ratio_h > UMBRAL_YAW:
            tipo = TIPO_PERFIL_D
        elif ratio_h < -UMBRAL_YAW:
            tipo = TIPO_PERFIL_I
        elif abs(ratio_v) > UMBRAL_ABAJO:
            tipo = TIPO_ABAJO
        else:
            tipo = TIPO_FRONTAL

        # Siempre imprimir (no solo cuando cambia) para ver valores en vivo
        barra_h = "#" * int(abs(ratio_h) * 50)
        signo_h = "+" if ratio_h >= 0 else "-"
        print(f"  {tipo:<14} {ratio_h:>+10.4f} {ratio_v:>+10.4f}  "
              f"{'SI':>6}  {signo_h}{barra_h[:20]}")

        time.sleep(0.15)

except KeyboardInterrupt:
    print("\n\n[Ctrl+C] Saliendo...")

finally:
    print(f"\nUmbrales usados:")
    print(f"  UMBRAL_YAW   = {UMBRAL_YAW:.3f}")
    print(f"  UMBRAL_ABAJO = {UMBRAL_ABAJO:.3f}")
    print("\nSi los laterales no se detectan, ajusta UMBRAL_YAW en face_engine.py")
    print("al valor que ves cuando volteas (por ejemplo 0.05 si ves +0.06)\n")
    if USAR_PICAM:
        picam2.close()
    else:
        cap.release()