"""
Corre este archivo para ver las distancias reales que calcula el sistema.
Apunta tu cara a la camara y te dira que distancia obtiene contra cada
persona registrada. Eso nos dice exactamente que umbral usar.
"""
import cv2
import numpy as np
import json, sqlite3, time
from picamera2 import Picamera2   # ← añadido

DB_PATH  = "reconocimiento_facial.db"
HAAR     = "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(HAAR)
FILAS, COLS = 8, 16
BLK_H, BLK_W = 128//FILAS, 128//COLS

def extraer_vector(frame):
    gris  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ecualizar histograma para mejor contraste
    gris  = cv2.equalizeHist(gris)
    caras = detector.detectMultiScale(gris, 1.1, 5, minSize=(80,80))
    if len(caras) == 0:
        return None, None
    x,y,w,h = sorted(caras, key=lambda c:c[2]*c[3], reverse=True)[0]
    cara = cv2.resize(gris[y:y+h, x:x+w], (128,128))
    v = []
    for i in range(FILAS):
        for j in range(COLS):
            b = cara[i*BLK_H:(i+1)*BLK_H, j*BLK_W:(j+1)*BLK_W]
            v.append(float(np.mean(b)))
    v = np.array(v, dtype=np.float32)
    v = (v - v.min()) / (v.max() - v.min() + 1e-6)
    return v, (x,y,w,h)

def cargar_bd():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT p.nombre_completo, p.numero_cuenta, vf.vector
        FROM vectores_faciales vf JOIN personas p ON p.id=vf.persona_id
    """).fetchall()
    conn.close()
    return [(n, c, np.array(json.loads(v), dtype=np.float32)) for n,c,v in rows]

registros = cargar_bd()
if not registros:
    print("[ERROR] No hay personas registradas en la base de datos.")
    exit()

print(f"\n[INFO] {len(registros)} persona(s) en la BD:")
for n,c,_ in registros:
    print(f"  - {n} ({c})")
print("\n[INFO] Abre la camara, apunta tu cara y mira las distancias...")
print("[INFO] Presiona Q para salir\n")

# ─── CAMBIO AQUÍ ─────────────────────────────
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "BGR888"}
))
picam2.start()
# ─────────────────────────────────────────────

muestras = []

while True:
    frame = picam2.capture_array()   # ← cambio (antes cap.read())
    ret = True

    if not ret:
        print("[WARN] No se pudo leer frame de la camara")
        continue

    frame = cv2.flip(frame, 1)
    v, coords = extraer_vector(frame)

    if v is not None:
        muestras.append(v)
        if len(muestras) >= 5:
            v_final = np.mean(muestras, axis=0).astype(np.float32)
            muestras = []
            print("--- Distancias calculadas ---")
            for nombre, cuenta, v_bd in registros:
                dist = float(np.linalg.norm(v_final - v_bd))
                estado = "RECONOCERIA" if dist <= 0.25 else (
                         "CERCA"       if dist <= 0.40 else
                         "LEJOS")
                print(f"  {nombre:30s} dist={dist:.4f}  [{estado}]")
            print()

        x,y,w,h = coords
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,212,255),2)
        cv2.putText(frame, f"Muestras: {len(muestras)}/5",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,212,255),2)

    cv2.imshow("Diagnostico - presiona Q para salir", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─── CAMBIO FINAL ────────────────────────────
picam2.close()   # ← antes era cap.release()
cv2.destroyAllWindows()