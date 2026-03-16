"""
diagnostico.py
==============
Herramienta de diagnostico para ver en tiempo real:
  - Que angulo esta detectando MediaPipe
  - Los valores exactos de yaw_ratio y pos_nariz_v
  - Las distancias contra personas registradas

Corre con:  python3 diagnostico.py
Presiona Q para salir, A/S para ajustar umbrales en vivo.
"""
import cv2
import numpy as np
import sys, os, time

# Asegurar que puede importar los modulos del proyecto
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

# ─── MediaPipe ────────────────────────────────────────────────────────────────
import mediapipe as mp
mp_face = mp.solutions.face_mesh
mesh   = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4)

# ─── Umbrales ajustables con teclas ──────────────────────────────────────────
UMBRAL_YAW    = 0.12
UMBRAL_ABAJO  = 0.34
UMBRAL_ARRIBA = 0.50

TIPOS = {
    "frontal":    (0,   212, 255),   # cyan
    "perfil_der": (0,   165, 255),   # naranja
    "perfil_izq": (255, 165, 0  ),   # azul
    "abajo":      (180, 0,   255),   # morado
    "sin_cara":   (80,  80,  80 ),   # gris
}

print("\n=== DIAGNOSTICO DE ANGULOS ===")
print("Teclas:")
print("  Q       — salir")
print("  Y/U     — bajar/subir umbral YAW")
print("  I/O     — bajar/subir umbral ABAJO")
print("  R       — reset umbrales")
print("  D       — mostrar distancias BD (5 muestras)")
print()

# ─── BD para distancias ───────────────────────────────────────────────────────
try:
    from database import cargar_vectores_por_angulo
    from face_engine import extraer_caracteristicas, distancia_ponderada
    registros_bd = cargar_vectores_por_angulo()
    print(f"[BD] {len(registros_bd)} vectores cargados de la BD")
    for r in registros_bd:
        print(f"     {r['nombre']} — angulo: {r['angulo']}")
except Exception as e:
    registros_bd = []
    print(f"[BD] No se pudo cargar BD: {e}")

muestras_dist = []
modo_distancias = False

# ─── loop principal ───────────────────────────────────────────────────────────
while True:
    frame = leer_frame()
    if frame is None:
        continue
    frame = cv2.flip(frame, 1)
    h_img, w_img = frame.shape[:2]

    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = mesh.process(rgb)

    tipo      = "sin_cara"
    yaw_ratio = 0.0
    pos_nariz = 0.0
    bbox      = None

    if resultado.multi_face_landmarks:
        lm = resultado.multi_face_landmarks[0].landmark

        def pt(idx):
            return np.array([lm[idx].x * w_img,
                             lm[idx].y * h_img,
                             lm[idx].z * w_img])

        nariz       = pt(1)
        ojo_izq     = pt(33)
        ojo_der     = pt(263)
        menton      = pt(152)
        frente      = pt(10)
        mejilla_izq = pt(234)
        mejilla_der = pt(454)

        centro_ojos = (ojo_izq + ojo_der) / 2.0

        dist_izq  = np.linalg.norm(nariz[:2] - mejilla_izq[:2])
        dist_der  = np.linalg.norm(nariz[:2] - mejilla_der[:2])
        total_d   = dist_izq + dist_der + 1e-6
        yaw_ratio = (dist_izq - dist_der) / total_d

        altura_cara = abs(menton[1] - frente[1]) + 1e-6
        pos_nariz   = (nariz[1] - centro_ojos[1]) / altura_cara

        # Clasificar con umbrales actuales
        if yaw_ratio > UMBRAL_YAW:
            tipo = "perfil_der"
        elif yaw_ratio < -UMBRAL_YAW:
            tipo = "perfil_izq"
        elif pos_nariz < UMBRAL_ABAJO:
            tipo = "abajo"
        elif pos_nariz > UMBRAL_ARRIBA:
            tipo = "abajo"
        else:
            tipo = "frontal"

        # Bounding box
        xs = [l.x * w_img for l in lm]
        ys = [l.y * h_img for l in lm]
        x1 = max(0, int(min(xs)) - 20)
        y1 = max(0, int(min(ys)) - 20)
        x2 = min(w_img, int(max(xs)) + 20)
        y2 = min(h_img, int(max(ys)) + 20)
        bbox = (x1, y1, x2, y2)

        # Acumular para distancias
        if modo_distancias and registros_bd:
            try:
                v, p, coords, t = extraer_caracteristicas(frame)
                if v is not None:
                    muestras_dist.append((v, p))
                    if len(muestras_dist) >= 5:
                        print("\n--- DISTANCIAS (5 muestras) ---")
                        v_final = np.mean([m[0] for m in muestras_dist], axis=0)
                        p_final = np.mean([m[1] for m in muestras_dist], axis=0)

                        # Agrupar por persona
                        por_persona = {}
                        for reg in registros_bd:
                            pid  = reg["persona_id"]
                            dist, nz = distancia_ponderada(v_final, p_final,
                                                            reg["vector"], reg["pesos"])
                            if pid not in por_persona or dist < por_persona[pid][0]:
                                por_persona[pid] = (dist, nz, reg["nombre"],
                                                    reg["angulo"])

                        for pid, (dist, nz, nombre, angulo) in por_persona.items():
                            estado = ("ACCESO" if dist <= 0.12 else
                                      "CERCA"  if dist <= 0.25 else "LEJOS")
                            print(f"  {nombre:25s} dist={dist:.4f} "
                                  f"angulo={angulo:12s} zonas={nz} [{estado}]")
                        print()
                        muestras_dist = []
                        modo_distancias = False
            except Exception as e:
                print(f"[ERR] {e}")

    # ── Dibujar ───────────────────────────────────────────────────────────────
    color = TIPOS[tipo]

    if bbox:
        x1, y1, x2, y2 = bbox
        L = max(15, (x2-x1)//4)
        for p1, p2 in [
            ((x1,y1),(x1+L,y1)), ((x1,y1),(x1,y1+L)),
            ((x2,y1),(x2-L,y1)), ((x2,y1),(x2,y1+L)),
            ((x1,y2),(x1+L,y2)), ((x1,y2),(x1,y2-L)),
            ((x2,y2),(x2-L,y2)), ((x2,y2),(x2,y2-L)),
        ]:
            cv2.line(frame, p1, p2, color, 2)

    # Panel de info
    panel_y = 20
    def txt(s, y, col=(220,220,220)):
        cv2.putText(frame, s, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

    txt(f"ANGULO:    {tipo.upper()}", panel_y,      color)
    txt(f"yaw_ratio: {yaw_ratio:+.3f}  (umbral +/-{UMBRAL_YAW:.2f})",
        panel_y+22)
    txt(f"pos_nariz: {pos_nariz:.3f}  "
        f"(abajo<{UMBRAL_ABAJO:.2f} arriba>{UMBRAL_ARRIBA:.2f})",
        panel_y+44)
    txt(f"YAW umbral: {UMBRAL_YAW:.3f}  [Y/U para ajustar]",
        panel_y+70, (160,160,160))
    txt(f"ABAJO umbral: {UMBRAL_ABAJO:.3f}  [I/O para ajustar]",
        panel_y+88, (160,160,160))

    if modo_distancias:
        txt(f"Midiendo distancias... {len(muestras_dist)}/5",
            panel_y+114, (0,255,136))

    cv2.imshow("Diagnostico de angulos — Q para salir", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('y'):
        UMBRAL_YAW = max(0.02, UMBRAL_YAW - 0.01)
        print(f"[YAW] umbral = {UMBRAL_YAW:.3f}")
    elif key == ord('u'):
        UMBRAL_YAW = min(0.40, UMBRAL_YAW + 0.01)
        print(f"[YAW] umbral = {UMBRAL_YAW:.3f}")
    elif key == ord('i'):
        UMBRAL_ABAJO = max(0.10, UMBRAL_ABAJO - 0.01)
        print(f"[ABAJO] umbral = {UMBRAL_ABAJO:.3f}")
    elif key == ord('o'):
        UMBRAL_ABAJO = min(0.60, UMBRAL_ABAJO + 0.01)
        print(f"[ABAJO] umbral = {UMBRAL_ABAJO:.3f}")
    elif key == ord('r'):
        UMBRAL_YAW   = 0.12
        UMBRAL_ABAJO = 0.34
        print("[RESET] Umbrales restaurados")
    elif key == ord('d'):
        modo_distancias = True
        muestras_dist   = []
        print("[DIST] Midiendo 5 muestras...")

# ─── Cierre ───────────────────────────────────────────────────────────────────
if USAR_PICAM:
    picam2.close()
else:
    cap.release()
cv2.destroyAllWindows()

print(f"\nUmbrales finales:")
print(f"  UMBRAL_YAW   = {UMBRAL_YAW:.3f}")
print(f"  UMBRAL_ABAJO = {UMBRAL_ABAJO:.3f}")
print("Copia estos valores a face_engine.py si los quieres guardar.")