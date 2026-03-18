"""
face_engine.py - VERSIÓN OPTIMIZADA PARA PERFILES
===================================================
Motor de reconocimiento facial multiángulo con LBP.

SOLUCIÓN: Umbrales MUCHO más bajos para YuNet en perfiles
          + Haar cascades mejorados como fallback principal
          + Mejor preprocesamiento de imagen
"""

import cv2
import numpy as np
import os

TIPO_FRONTAL  = "frontal"
TIPO_PERFIL_D = "perfil_der"
TIPO_PERFIL_I = "perfil_izq"
TIPO_ABAJO    = "abajo"

_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

# =============================================================================
#  DETECCION
# =============================================================================

_yunet         = None
_dnn_net       = None
_haar_frontal  = None
_haar_perfil   = None
_dlib_detector = None
_det_init      = False


def _init_detectores():
    """Inicializa detectores."""
    global _yunet, _dnn_net, _haar_frontal, _haar_perfil, _det_init
    if _det_init:
        return
    _det_init = True

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

    # ── YuNet ──────────────────────────────────────────────────────────────────
    yunet_ruta = os.path.join(base, "face_detection_yunet.onnx")
    if os.path.exists(yunet_ruta):
        try:
            _yunet = cv2.FaceDetectorYN.create(
                yunet_ruta, "", (640, 480),
                score_threshold=0.1, nms_threshold=0.3, top_k=20)
            print(f"[DET] YuNet: {yunet_ruta}")
        except Exception as e:
            print(f"[DET] YuNet error: {e}")
    else:
        print("[DET] YuNet no encontrado")

    # ── DNN SSD ────────────────────────────────────────────────────────────────
    proto = os.path.join(base, "opencv_face_detector.prototxt")
    caffe = os.path.join(base, "opencv_face_detector.caffemodel")
    if os.path.exists(proto) and os.path.exists(caffe):
        try:
            _dnn_net = cv2.dnn.readNetFromCaffe(proto, caffe)
            print(f"[DET] DNN SSD: {caffe}")
        except Exception as e:
            print(f"[DET] DNN SSD error: {e}")
    else:
        print("[DET] DNN SSD no encontrado")

    # ── Haar frontal ───────────────────────────────────────────────────────────
    rutas_frontal = [
        os.path.join(base, "haarcascade_frontalface_default.xml"),
    ]
    try:
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            rutas_frontal.append(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except:
        pass
    
    for ruta in rutas_frontal:
        if os.path.exists(ruta):
            _haar_frontal = cv2.CascadeClassifier(ruta)
            if not _haar_frontal.empty():
                print(f"[DET] Haar frontal: {ruta}")
                break

    # ── Haar perfil ────────────────────────────────────────────────────────────
    ruta_perfil = None
    try:
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            ruta_perfil = cv2.data.haarcascades + "haarcascade_profileface.xml"
    except:
        pass
    
    if ruta_perfil and os.path.exists(ruta_perfil):
        _haar_perfil = cv2.CascadeClassifier(ruta_perfil)
        if not _haar_perfil.empty():
            print(f"[DET] Haar perfil: {ruta_perfil}")
        else:
            _haar_perfil = None
    else:
        print("[DET] Haar perfil no encontrado - usará Haar frontal relajado")

    # ── Dlib ───────────────────────────────────────────────────────────────────
    try:
        import dlib
        _dlib_detector = dlib.get_frontal_face_detector()
        print("[DET] Dlib: disponible")
    except ImportError:
        _dlib_detector = None


def _preprocess_frame(frame):
    """Preprocesa el frame para mejorar detección de perfiles."""
    # Mejorar contraste
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _detectar_dnn(frame, tipo_esperado=None):
    """
    Detecta caras con énfasis en perfiles.
    Orden: YuNet (ultra-sensible) → DNN → Haar perfil → Haar frontal relajado → Dlib
    """
    _init_detectores()
    h_img, w_img = frame.shape[:2]
    resultados   = []
    es_perfil = tipo_esperado in (TIPO_PERFIL_D, TIPO_PERFIL_I)

    # Preprocesar frame
    frame_prep = _preprocess_frame(frame)

    # ── YuNet (ULTRA SENSIBLE PARA PERFILES) ───────────────────────────────────
    if _yunet is not None:
        try:
            _yunet.setInputSize((w_img, h_img))
            
            # Parámetros MUCHO más permisivos para perfiles
            if es_perfil:
                _yunet.setScoreThreshold(0.05)  # ANTES: 0.4 → AHORA: 0.05
                _yunet.setTopK(30)              # ANTES: 5   → AHORA: 30
            else:
                _yunet.setScoreThreshold(0.2)   # ANTES: 0.4 → AHORA: 0.2
                _yunet.setTopK(10)
            
            _, faces = _yunet.detect(frame_prep)
            
            if faces is not None and len(faces) > 0:
                for face in faces:
                    x  = max(0, int(face[0]))
                    y  = max(0, int(face[1]))
                    w  = int(face[2])
                    h  = int(face[3])
                    cf = float(face[14])
                    
                    # Filtro de tamaño más permisivo
                    if w > 15 and h > 15:
                        resultados.append((x, y, min(w_img-x, w), min(h_img-y, h), cf))
                
                if resultados:
                    resultados.sort(key=lambda c: c[2]*c[3], reverse=True)
                    print(f"[YuNet] Detectadas {len(resultados)} cara(s), confianza: {resultados[0][4]:.2f}")
                    return resultados
        except Exception as e:
            print(f"[YuNet] Error: {e}")

    # ── DNN SSD ────────────────────────────────────────────────────────────────
    if _dnn_net is not None:
        try:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame_prep, (300, 300)), 1.0,
                (300, 300), (104.0, 117.0, 123.0))
            _dnn_net.setInput(blob)
            dets = _dnn_net.forward()
            
            # Umbrales muy bajos para perfiles
            umbral_conf = 0.1 if es_perfil else 0.35
            
            for i in range(dets.shape[2]):
                conf = float(dets[0, 0, i, 2])
                if conf < umbral_conf:
                    continue
                x1 = max(0, int(dets[0, 0, i, 3] * w_img))
                y1 = max(0, int(dets[0, 0, i, 4] * h_img))
                x2 = min(w_img, int(dets[0, 0, i, 5] * w_img))
                y2 = min(h_img, int(dets[0, 0, i, 6] * h_img))
                if x2 > x1 and y2 > y1:
                    resultados.append((x1, y1, x2-x1, y2-y1, conf))
            
            if resultados:
                resultados.sort(key=lambda c: c[2]*c[3], reverse=True)
                print(f"[DNN SSD] Detectadas {len(resultados)} cara(s)")
                return resultados
        except Exception as e:
            print(f"[DNN SSD] Error: {e}")

    # ── Haar Cascades (PRINCIPAL PARA PERFILES) ────────────────────────────────
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)
    
    # Aplicar CLAHE para mejorar contraste en perfiles
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gris = clahe.apply(gris)

    # Haar perfil
    if es_perfil and _haar_perfil is not None:
        try:
            if tipo_esperado == TIPO_PERFIL_I:
                gris_det = cv2.flip(gris, 1)
            else:
                gris_det = gris

            # Parámetros RELAJADOS para detectar perfiles completos
            caras = _haar_perfil.detectMultiScale(
                gris_det, 
                scaleFactor=1.02,    # ANTES: 1.05 → AHORA: 1.02 (más sensible)
                minNeighbors=2,      # ANTES: 3    → AHORA: 2
                minSize=(30, 30),    # ANTES: 40   → AHORA: 30
                maxSize=(300, 300))

            if len(caras) > 0:
                for (x, y, w, h) in caras:
                    if tipo_esperado == TIPO_PERFIL_I:
                        x = w_img - x - w
                    resultados.append((int(x), int(y), int(w), int(h), 0.9))
                
                resultados.sort(key=lambda c: c[2]*c[3], reverse=True)
                print(f"[Haar Perfil] Detectadas {len(caras)} cara(s)")
                return resultados
        except Exception as e:
            print(f"[Haar Perfil] Error: {e}")

    # Haar frontal RELAJADO (para cuando el perfil es parcial)
    if _haar_frontal is not None:
        try:
            if es_perfil:
                # Parámetros MÁS relajados para perfiles
                scale, vecinos, tam = 1.02, 2, (30, 30)
            else:
                scale, vecinos, tam = 1.1, 5, (60, 60)

            caras = _haar_frontal.detectMultiScale(
                gris, scaleFactor=scale, minNeighbors=vecinos,
                minSize=tam, flags=cv2.CASCADE_SCALE_IMAGE)

            if len(caras) > 0:
                for (x, y, w, h) in caras:
                    resultados.append((int(x), int(y), int(w), int(h), 0.8))
                
                resultados.sort(key=lambda c: c[2]*c[3], reverse=True)
                print(f"[Haar Frontal] Detectadas {len(caras)} cara(s) (modo relajado)")
                return resultados
        except Exception as e:
            print(f"[Haar Frontal] Error: {e}")

    # ── Dlib (último recurso) ──────────────────────────────────────────────────
    if not resultados and _dlib_detector is not None:
        try:
            dets = _dlib_detector(gris, 1)
            for det in dets:
                x = max(0, det.left())
                y = max(0, det.top())
                w = det.right() - det.left()
                h = det.bottom() - det.top()
                if w > 20 and h > 20:
                    resultados.append((x, y, w, h, 0.7))
            
            if resultados:
                print(f"[Dlib] Detectadas {len(resultados)} cara(s)")
        except Exception as e:
            print(f"[Dlib] Error: {e}")

    return resultados


# =============================================================================
#  CLASIFICACION DE ANGULO
# =============================================================================

_lm_det    = None
_buf_yaw   = []
_BUF_N_YAW = 3
_yunet_kpts = None


def _get_lm_detector():
    """Carga detector LBF."""
    global _lm_det
    if _lm_det is not None:
        return _lm_det if _lm_det is not False else None
    
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "models", "lbfmodel.yaml")
    if os.path.exists(ruta):
        try:
            det = cv2.face.createFacemarkLBF()
            det.loadModel(ruta)
            _lm_det = det
            print(f"[LBF] Cargado: {ruta}")
            return _lm_det
        except Exception as e:
            print(f"[LBF] Error: {e}")
    
    _lm_det = False
    print("[LBF] No disponible, usando Sobel")
    return None


def _calcular_yaw_lbf(frame_gris, bbox):
    """Calcula yaw con LBF landmarks."""
    lm_det = _get_lm_detector()
    if lm_det is None:
        return None
    
    x, y, w, h = bbox
    try:
        rect = np.array([[x, y, w, h]], dtype=np.int32)
        ok, lms = lm_det.fit(frame_gris, rect)
        if not ok or len(lms) == 0:
            return None
        
        lm      = lms[0][0]
        nariz   = lm[30].astype(np.float64)
        ojo_izq = lm[36].astype(np.float64)
        ojo_der = lm[45].astype(np.float64)
        
        d_izq   = float(np.linalg.norm(nariz - ojo_izq))
        d_der   = float(np.linalg.norm(nariz - ojo_der))
        total   = d_izq + d_der + 1e-6
        
        return -((d_der - d_izq) / total) * 120.0
    except:
        return None


def _calcular_yaw_sobel(frame_gris, bbox, frame_shape):
    """Calcula yaw con Sobel."""
    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_shape[1], x + w)
    y2 = min(frame_shape[0], y + h)
    
    recorte = frame_gris[y1:y2, x1:x2]
    if recorte.size == 0:
        return 0.0
    
    try:
        cara = cv2.resize(recorte, (128, 128))
        gx   = cv2.Sobel(cara, cv2.CV_32F, 1, 0, ksize=5)
        gy   = cv2.Sobel(cara, cv2.CV_32F, 0, 1, ksize=5)
        b    = np.sqrt(gx**2 + gy**2)
        
        w_   = b.shape[1]
        m    = int(w_ * 0.12)
        izq  = float(np.mean(b[:, :w_//2 - m]))
        der  = float(np.mean(b[:, w_//2 + m:]))
        
        return -((der - izq) / (der + izq + 1e-6)) * 120.0
    except:
        return 0.0


def _clasificar_angulo(frame_gris, bbox, frame_shape, tipo_esperado=None):
    """Clasifica el ángulo."""
    global _buf_yaw

    yaw = None
    yaw = _calcular_yaw_lbf(frame_gris, bbox)
    
    if yaw is None:
        yaw = _calcular_yaw_sobel(frame_gris, bbox, frame_shape)

    _buf_yaw.append(yaw)
    if len(_buf_yaw) > _BUF_N_YAW:
        _buf_yaw.pop(0)

    if _buf_yaw:
        yaw_s = float(_buf_yaw[int(np.argmax(np.abs(_buf_yaw)))])
    else:
        yaw_s = yaw

    umbral = 8.0 if tipo_esperado in (TIPO_PERFIL_D, TIPO_PERFIL_I) else 15.0

    if   yaw_s >  umbral: return TIPO_PERFIL_D
    elif yaw_s < -umbral: return TIPO_PERFIL_I
    else:                 return TIPO_FRONTAL


def _extraer_angulos_lbf(gris, bbox, fw, fh):
    """Extrae yaw y pitch."""
    yaw = _calcular_yaw_lbf(gris, bbox)
    pitch = 0.0
    return yaw, pitch


# =============================================================================
#  ZONAS LBP
# =============================================================================

ZONAS_FRONTAL = [
    (0,  40,  0,  128, "frente"),
    (28, 65,  0,   58, "ojo_izq"),
    (28, 65, 70,  128, "ojo_der"),
    (55, 92, 28,  100, "nariz"),
    (62, 100, 0,   50, "mejilla_izq"),
    (62, 100, 78, 128, "mejilla_der"),
    (88, 128, 14, 114, "boca_menton"),
    (20, 100,  0, 128, "cara_media"),
]

ZONAS_PERFIL_D = [
    (0,  40,  0,  128, "frente"),
    (20, 60,  0,   60, "ojo_izq"),
    (45, 85,  0,   65, "nariz_lat"),
    (55, 95,  0,   55, "mejilla_izq"),
    (65, 110, 0,   50, "mandibula"),
    (25, 75,  0,   40, "pomulo"),
    (82, 128, 0,   75, "menton"),
    (15, 105, 0,   80, "perfil_media"),
]

ZONAS_PERFIL_I = [
    (0,  40,  0,  128, "frente"),
    (20, 60,  68, 128, "ojo_der"),
    (45, 85,  63, 128, "nariz_lat"),
    (55, 95,  73, 128, "mejilla_der"),
    (65, 110, 78, 128, "mandibula"),
    (25, 75,  88, 128, "pomulo"),
    (82, 128, 53, 128, "menton"),
    (15, 105, 48, 128, "perfil_media"),
]

ZONAS_ABAJO = ZONAS_FRONTAL

ZONAS_POR_TIPO = {
    TIPO_FRONTAL:  ZONAS_FRONTAL,
    TIPO_PERFIL_D: ZONAS_PERFIL_D,
    TIPO_PERFIL_I: ZONAS_PERFIL_I,
    TIPO_ABAJO:    ZONAS_FRONTAL,
}

ZONAS      = ZONAS_FRONTAL
N_ZONAS    = 8
LBP_BINS   = 64
VECTOR_DIM = N_ZONAS * LBP_BINS


# =============================================================================
#  LBP
# =============================================================================

_UNIFORM_MAP = None


def _build_uniform_map():
    umap = np.full(256, 58, dtype=np.int32)
    idx  = 0
    for code in range(256):
        b = format(code, '08b')
        t = sum(b[i] != b[(i+1) % 8] for i in range(8))
        if t <= 2:
            umap[code] = idx
            idx += 1
    return umap


def _get_uniform_map():
    global _UNIFORM_MAP
    if _UNIFORM_MAP is None:
        _UNIFORM_MAP = _build_uniform_map()
    return _UNIFORM_MAP


def _lbp_imagen(gris):
    img    = gris.astype(np.int32)
    pad    = np.pad(img, 1, mode='edge')
    h, w   = gris.shape
    center = pad[1:-1, 1:-1]
    nbrs   = [
        pad[0:-2, 0:-2], pad[0:-2, 1:-1], pad[0:-2, 2:],
        pad[1:-1, 2:],   pad[2:,   2:],   pad[2:,   1:-1],
        pad[2:,   0:-2], pad[1:-1, 0:-2],
    ]
    lbp = np.zeros((h, w), dtype=np.uint8)
    for bit, nbr in enumerate(nbrs):
        lbp |= ((nbr >= center).astype(np.uint8) << bit)
    return lbp


def _histograma_zona(cara128, r0, r1, c0, c1):
    zona = cara128[r0:r1, c0:c1]
    if zona.size == 0:
        return np.zeros(LBP_BINS, dtype=np.float32)
    lbp_map = _lbp_imagen(zona)
    umap    = _get_uniform_map()
    hist59  = np.bincount(umap[lbp_map.flatten()],
                          minlength=59).astype(np.float32)
    total = hist59.sum()
    if total > 0:
        hist59 /= total
    hist64      = np.zeros(LBP_BINS, dtype=np.float32)
    hist64[:59] = hist59
    return hist64


# =============================================================================
#  API PÚBLICA
# =============================================================================

def preprocesar_cara(gris_zona):
    """Preprocesa zona de cara."""
    return cv2.GaussianBlur(_clahe.apply(gris_zona), (3, 3), 0)


def extraer_caracteristicas(frame, haar_path=None, modo="auto",
                             tipo_esperado=None):
    """Detecta cara y extrae vector LBP."""
    caras = _detectar_dnn(frame, tipo_esperado=tipo_esperado)
    if not caras:
        return None, None, None

    x, y, w, h, _ = caras[0]
    h_img, w_img  = frame.shape[:2]
    x1, y1        = max(0, x), max(0, y)
    x2, y2        = min(w_img, x + w), min(h_img, y + h)

    gris_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recorte   = preprocesar_cara(gris_full[y1:y2, x1:x2])
    if recorte.size == 0:
        return None, None, None

    cara128 = cv2.resize(recorte, (128, 128))

    tipo = _clasificar_angulo(
        gris_full, (x1, y1, x2-x1, y2-y1),
        frame.shape, tipo_esperado=tipo_esperado)

    zonas  = ZONAS_POR_TIPO.get(tipo, ZONAS_FRONTAL)
    vector = np.concatenate([
        _histograma_zona(cara128, r0, r1, c0, c1)
        for r0, r1, c0, c1, _ in zonas
    ]).astype(np.float32)

    return vector, (x1, y1, x2-x1, y2-y1), tipo


def distancia_chi2(v1, v2):
    """Distancia chi-cuadrado."""
    denom = v1 + v2 + 1e-7
    return float(np.sum((v1 - v2) ** 2 / denom))


def dibujar_overlay(frame, coords, color, texto="", tipo=None):
    """Dibuja overlay."""
    x, y, w, h = coords
    L = max(18, w // 4)

    colores_tipo = {
        TIPO_FRONTAL:  (0,   212, 255),
        TIPO_PERFIL_D: (255, 165,   0),
        TIPO_PERFIL_I: (0,   165, 255),
        TIPO_ABAJO:    (180,   0, 255),
    }
    c = colores_tipo.get(tipo, color)

    for p1, p2 in [
        ((x,     y),   (x+L,   y)),   ((x,     y),   (x,     y+L)),
        ((x+w,   y),   (x+w-L, y)),   ((x+w,   y),   (x+w,   y+L)),
        ((x,     y+h), (x+L,   y+h)), ((x,     y+h), (x,     y+h-L)),
        ((x+w,   y+h), (x+w-L, y+h)), ((x+w,   y+h), (x+w,   y+h-L)),
    ]:
        cv2.line(frame, p1, p2, c, 2)

    etiquetas = {
        TIPO_FRONTAL:  "FRONTAL",
        TIPO_PERFIL_D: "PERFIL DER",
        TIPO_PERFIL_I: "PERFIL IZQ",
    }
    if tipo in etiquetas:
        cv2.putText(frame, etiquetas[tipo], (x, y+h+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    if texto:
        cv2.putText(frame, texto, (x, max(14, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame