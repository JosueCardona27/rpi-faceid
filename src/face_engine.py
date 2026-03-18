"""
face_engine.py - VERSIÓN MEJORADA CON MEDIAPIPE
==================================================
Motor de reconocimiento facial multiángulo con LBP.

Detección (en orden):
  1. MediaPipe    — detector robusto (±90°)
  2. YuNet        — si existe models/face_detection_yunet.onnx
  3. DNN SSD      — si existe models/opencv_face_detector.caffemodel
  4. Haar         — cascadas de fallback
  5. Dlib         — último recurso

Mejoras:
  - MediaPipe detecta perfiles completos (±90°)
  - Umbrales adaptativos por contexto
  - Mejor manejo de excepciones
  - Logging detallado para debugging
"""

import cv2
import numpy as np
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TIPO_FRONTAL  = "frontal"
TIPO_PERFIL_D = "perfil_der"
TIPO_PERFIL_I = "perfil_izq"
TIPO_ABAJO    = "abajo"

_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

# =============================================================================
#  DETECCION
# =============================================================================

_mediapipe_detector = None
_yunet              = None
_dnn_net            = None
_haar_frontal       = None
_haar_perfil        = None
_dlib_detector      = None
_det_init           = False


def _init_detectores():
    """Inicializa todos los detectores disponibles."""
    global _mediapipe_detector, _yunet, _dnn_net, _haar_frontal, _haar_perfil, _dlib_detector, _det_init
    
    if _det_init:
        return
    _det_init = True

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

    # ── MediaPipe (PRINCIPAL) ──────────────────────────────────────────────────
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        _mediapipe_detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 1=full range, 0=short range
            min_detection_confidence=0.3
        )
        logger.info("[DET] MediaPipe: disponible (PRINCIPAL)")
    except ImportError:
        logger.warning("[DET] MediaPipe no instalado. Instalar: pip install mediapipe")
        _mediapipe_detector = None

    # ── YuNet ──────────────────────────────────────────────────────────────────
    yunet_ruta = os.path.join(base, "face_detection_yunet.onnx")
    if os.path.exists(yunet_ruta):
        try:
            _yunet = cv2.FaceDetectorYN.create(
                yunet_ruta, "", (640, 480),
                score_threshold=0.3, nms_threshold=0.3, top_k=10)
            logger.info(f"[DET] YuNet: {yunet_ruta}")
        except Exception as e:
            logger.error(f"[DET] YuNet error: {e}")
    else:
        logger.info("[DET] YuNet no encontrado")

    # ── DNN SSD ──────────────────────────────────────────────────���─────────────
    proto = os.path.join(base, "opencv_face_detector.prototxt")
    caffe = os.path.join(base, "opencv_face_detector.caffemodel")
    if os.path.exists(proto) and os.path.exists(caffe):
        try:
            _dnn_net = cv2.dnn.readNetFromCaffe(proto, caffe)
            logger.info(f"[DET] DNN SSD: {caffe}")
        except Exception as e:
            logger.error(f"[DET] DNN SSD error: {e}")
    else:
        logger.info("[DET] DNN SSD no encontrado")

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
            logger.info(f"[DET] Haar frontal: {ruta}")
            break

    if _haar_frontal is None or _haar_frontal.empty():
        logger.warning("[DET] Haar frontal no cargado correctamente")
        _haar_frontal = None

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
            logger.info(f"[DET] Haar perfil: {ruta_perfil}")
        else:
            _haar_perfil = None
            logger.warning("[DET] Haar perfil no cargado correctamente")
    else:
        logger.info("[DET] Haar perfil no encontrado")

    # ── Dlib ───────────────────────────────────────────────────────────────────
    try:
        import dlib
        _dlib_detector = dlib.get_frontal_face_detector()
        logger.info("[DET] Dlib: disponible")
    except ImportError:
        logger.info("[DET] Dlib no instalado")
        _dlib_detector = None


def _detectar_mediapipe(frame):
    """Detecta caras usando MediaPipe."""
    if _mediapipe_detector is None:
        return []
    
    try:
        h_img, w_img = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = _mediapipe_detector.process(rgb_frame)
        resultados = []
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w_img))
                y = max(0, int(bbox.ymin * h_img))
                w = int(bbox.width * w_img)
                h = int(bbox.height * h_img)
                conf = detection.score[0]
                
                if w > 20 and h > 20:
                    resultados.append((x, y, w, h, conf))
        
        return sorted(resultados, key=lambda c: c[2]*c[3], reverse=True)
    except Exception as e:
        logger.error(f"[MediaPipe] Error: {e}")
        return []


def _detectar_dnn(frame, tipo_esperado=None):
    """
    Detecta la cara principal segun el angulo esperado.
    Orden: MediaPipe → YuNet → DNN SSD → Haar → Dlib
    """
    _init_detectores()
    h_img, w_img = frame.shape[:2]
    resultados   = []
    es_perfil = tipo_esperado in (TIPO_PERFIL_D, TIPO_PERFIL_I)

    # ── MediaPipe (PRINCIPAL) ──────────────────────────────────────────────────
    resultados = _detectar_mediapipe(frame)
    if resultados:
        logger.debug(f"[MediaPipe] Detectadas {len(resultados)} cara(s)")
        return resultados

    # ── YuNet ──────────────────────────────────────────────────────────────────
    if _yunet is not None:
        try:
            if es_perfil:
                _yunet.setScoreThreshold(0.25)
                _yunet.setTopK(15)
            else:
                _yunet.setScoreThreshold(0.35)
                _yunet.setTopK(5)
            
            _yunet.setInputSize((w_img, h_img))
            _, faces = _yunet.detect(frame)
            
            if faces is not None:
                for face in faces:
                    x  = max(0, int(face[0]))
                    y  = max(0, int(face[1]))
                    w  = int(face[2])
                    h  = int(face[3])
                    cf = float(face[14])
                    if cf >= (0.25 if es_perfil else 0.35) and w > 20 and h > 20:
                        resultados.append((x, y, min(w_img-x, w), min(h_img-y, h), cf))
            
            if resultados:
                logger.debug(f"[YuNet] Detectadas {len(resultados)} cara(s)")
                return sorted(resultados, key=lambda c: c[2]*c[3], reverse=True)
        except Exception as e:
            logger.error(f"[YuNet] Error: {e}")

    # ── DNN SSD ────────────────────────────────────────────────────────────────
    if _dnn_net is not None:
        try:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 117.0, 123.0))
            _dnn_net.setInput(blob)
            dets = _dnn_net.forward()
            umbral_conf = 0.2 if es_perfil else 0.45
            
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
                logger.debug(f"[DNN SSD] Detectadas {len(resultados)} cara(s)")
                return sorted(resultados, key=lambda c: c[2]*c[3], reverse=True)
        except Exception as e:
            logger.error(f"[DNN SSD] Error: {e}")

    # ── Haar Cascades ──────────────────────────────────────────────────────────
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)

    if es_perfil and _haar_perfil is not None:
        try:
            if tipo_esperado == TIPO_PERFIL_I:
                gris_det = cv2.flip(gris, 1)
            else:
                gris_det = gris

            caras = _haar_perfil.detectMultiScale(
                gris_det, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))

            if len(caras) > 0:
                for (x, y, w, h) in caras:
                    if tipo_esperado == TIPO_PERFIL_I:
                        x = w_img - x - w
                    resultados.append((int(x), int(y), int(w), int(h), 0.85))
                logger.debug(f"[Haar Perfil] Detectadas {len(caras)} cara(s)")
                return sorted(resultados, key=lambda c: c[2]*c[3], reverse=True)
        except Exception as e:
            logger.error(f"[Haar Perfil] Error: {e}")

    if _haar_frontal is not None:
        try:
            if es_perfil:
                scale, vecinos, tam = 1.05, 3, (40, 40)
            else:
                scale, vecinos, tam = 1.1, 5, (60, 60)

            caras = _haar_frontal.detectMultiScale(
                gris, scaleFactor=scale, minNeighbors=vecinos,
                minSize=tam, flags=cv2.CASCADE_SCALE_IMAGE)

            if len(caras) > 0:
                for (x, y, w, h) in caras:
                    resultados.append((int(x), int(y), int(w), int(h), 0.8))
                logger.debug(f"[Haar Frontal] Detectadas {len(caras)} cara(s)")
                return sorted(resultados, key=lambda c: c[2]*c[3], reverse=True)
        except Exception as e:
            logger.error(f"[Haar Frontal] Error: {e}")

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
                logger.debug(f"[Dlib] Detectadas {len(resultados)} cara(s)")
        except Exception as e:
            logger.error(f"[Dlib] Error: {e}")

    return resultados


# =============================================================================
#  CLASIFICACION DE ANGULO
# =============================================================================

_lm_det    = None
_buf_yaw   = []
_BUF_N_YAW = 3
_yunet_kpts = None


def _get_lm_detector():
    """Carga el detector de landmarks LBF."""
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
            logger.info(f"[LBF] Cargado: {ruta}")
            return _lm_det
        except Exception as e:
            logger.error(f"[LBF] Error: {e}")
    
    _lm_det = False
    logger.info("[LBF] No disponible, usando Sobel")
    return None


def _calcular_yaw_lbf(frame_gris, bbox):
    """Calcula yaw usando LBF landmarks."""
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
    except Exception as e:
        logger.debug(f"[LBF] Error en cálculo: {e}")
        return None


def _calcular_yaw_sobel(frame_gris, bbox, frame_shape):
    """Calcula yaw usando análisis Sobel."""
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
        b    = np.sqrt(gx**2 + gy**2)  # magnitud euclidiana
        
        w_   = b.shape[1]
        m    = int(w_ * 0.12)
        izq  = float(np.mean(b[:, :w_//2 - m]))
        der  = float(np.mean(b[:, w_//2 + m:]))
        
        return -((der - izq) / (der + izq + 1e-6)) * 120.0
    except Exception as e:
        logger.debug(f"[Sobel] Error: {e}")
        return 0.0


def _clasificar_angulo(frame_gris, bbox, frame_shape, tipo_esperado=None):
    """Clasifica el ángulo de la cara."""
    global _buf_yaw, _yunet_kpts

    yaw = None
    
    # 1. LBF landmarks
    yaw = _calcular_yaw_lbf(frame_gris, bbox)
    
    # 2. Sobel (fallback)
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
    """Extrae yaw y pitch usando LBF."""
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
    """Preprocesa la zona de la cara."""
    return cv2.GaussianBlur(_clahe.apply(gris_zona), (3, 3), 0)


def extraer_caracteristicas(frame, haar_path=None, modo="auto",
                             tipo_esperado=None):
    """
    Detecta cara y extrae vector LBP de 512 dims.
    Retorna (vector, coords, tipo) o (None, None, None).
    """
    caras = _detectar_dnn(frame, tipo_esperado=tipo_esperado)
    if not caras:
        logger.debug("No se detectó cara")
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
    """Calcula distancia chi-cuadrado entre vectores."""
    denom = v1 + v2 + 1e-7
    return float(np.sum((v1 - v2) ** 2 / denom))


def dibujar_overlay(frame, coords, color, texto="", tipo=None):
    """Dibuja el overlay de detección en el frame."""
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
        TIPO_ABAJO:    "ABAJO",
    }
    if tipo in etiquetas:
        cv2.putText(frame, etiquetas[tipo], (x, y+h+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    if texto:
        cv2.putText(frame, texto, (x, max(14, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame