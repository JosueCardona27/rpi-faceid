"""
test_angulos.py - muestra yaw y pitch exactos en grados
Corre: python3 test_angulos.py
"""
import cv2, numpy as np, os, time, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from picamera2 import Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640,480), "format": "BGR888"}))
picam2.start()
time.sleep(0.5)

ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "models", "lbfmodel.yaml")
det = cv2.face.createFacemarkLBF()
det.loadModel(ruta)

from face_engine import _detectar_dnn

_PTS_3D = np.array([
    [ 0.0,   0.0,   0.0 ],
    [ 0.0, -63.6, -12.5 ],
    [-43.3, 32.7, -26.0 ],
    [ 43.3, 32.7, -26.0 ],
    [-28.9,-28.9, -24.1 ],
    [ 28.9,-28.9, -24.1 ],
], dtype=np.float64)
IDX = [30, 8, 36, 45, 48, 54]

print("\nVoltea la cabeza — Ctrl+C para salir")
print(f"{'TIPO':<14} {'YAW':>8} {'PITCH':>8}  (grados)")
print("-"*40)

try:
    while True:
        frame = cv2.flip(picam2.capture_array(), 1)
        caras = _detectar_dnn(frame)
        if not caras:
            time.sleep(0.2)
            continue

        x,y,w,h,_ = caras[0]
        fw, fh = frame.shape[1], frame.shape[0]
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect = np.array([[x,y,w,h]], dtype=np.int32)
        ok, lm = det.fit(gris, rect)

        if not ok or len(lm) == 0:
            print("LBF fallo")
            time.sleep(0.2)
            continue

        pts = np.array([lm[0][0][i] for i in IDX], dtype=np.float64)
        cam = np.array([[fw,0,fw/2],[0,fw,fh/2],[0,0,1]], dtype=np.float64)
        ok2, rvec, _ = cv2.solvePnP(_PTS_3D, pts, cam,
                                     np.zeros((4,1)),
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok2:
            time.sleep(0.2)
            continue

        rmat, _ = cv2.Rodrigues(rvec)
        yaw   = float(np.degrees(np.arctan2(rmat[1,0], rmat[0,0])))
        pitch = float(np.degrees(np.arctan2(-rmat[2,0],
                      np.sqrt(rmat[2,1]**2 + rmat[2,2]**2))))

        if yaw > 20:      tipo = "perfil_der"
        elif yaw < -20:   tipo = "perfil_izq"
        elif abs(pitch)>18: tipo = "abajo"
        else:             tipo = "frontal"

        print(f"{tipo:<14} {yaw:>+8.1f} {pitch:>+8.1f}")
        time.sleep(0.2)

except KeyboardInterrupt:
    pass
finally:
    picam2.close()