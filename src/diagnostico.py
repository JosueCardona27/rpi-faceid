"""
diagnostico.py - muestra el angulo REAL que calcula face_engine
Corre: python3 diagnostico.py
Ctrl+C para salir
"""
import cv2, numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from picamera2 import Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "BGR888"}))
picam2.start()
time.sleep(0.5)

from face_engine import (_detectar_dnn, _clasificar_angulo,
                          _extraer_angulos_lbf,
                          TIPO_FRONTAL, TIPO_PERFIL_D,
                          TIPO_PERFIL_I, TIPO_ABAJO)

print("\n" + "="*60)
print("  DIAGNOSTICO REAL — usando face_engine._clasificar_angulo")
print("="*60)
print(f"  {'TIPO':<14} {'YAW':>8} {'PITCH':>8}  cara")
print("  " + "-"*45)

n = 0
try:
    while True:
        frame = cv2.flip(picam2.capture_array(), 1)
        n += 1
        if n % 3 != 0:
            time.sleep(0.03)
            continue

        caras = _detectar_dnn(frame)
        if not caras:
            print(f"  {'SIN CARA':<14} {'---':>8} {'---':>8}")
            time.sleep(0.15)
            continue

        x, y, w, h, _ = caras[0]
        gris  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox  = (x, y, w, h)
        fw, fh = frame.shape[1], frame.shape[0]

        yaw, pitch = _extraer_angulos_lbf(gris, bbox, fw, fh)
        tipo = _clasificar_angulo(gris, bbox, frame.shape)

        if yaw is not None:
            print(f"  {tipo:<14} {yaw:>+8.1f} {pitch:>+8.1f}  SI")
        else:
            print(f"  {tipo:<14} {'LBF fallo':>8} {'':>8}  SI (asimetria)")
        time.sleep(0.15)

except KeyboardInterrupt:
    print("\n[Ctrl+C] Saliendo...")
finally:
    picam2.close()