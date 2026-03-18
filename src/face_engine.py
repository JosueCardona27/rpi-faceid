"""
face_engine.py
==============
Motor de reconocimiento facial multiangulo con LBP.

Deteccion (en orden):
  1. YuNet     — si existe models/face_detection_yunet.onnx
  2. DNN SSD   — si existe models/opencv_face_detector.caffemodel
  3. Haar      — frontal para pasos FRONTAL, perfil para pasos de lado

NOTA IMPORTANTE sobre el flip:
  interfaz.py voltea el frame con cv2.flip(raw, 1) antes de almacenarlo.
  El Sobel opera sobre el frame ya volteado, por lo que el yaw debe
  negarse para que DERECHA/IZQUIERDA sean correctos.

Vectores: LBP uniforme (59 bins + 5 padding = 64) x 8 zonas = 512 dims
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
_det_init      = False


def _init_detectores():
    global _yunet, _dnn_net, _haar_frontal, _haar_perfil, _det_init
    if _det_init:
        return
    _det_init = True

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

    # ── YuNet ─────────────────────────────────────────────────────────────────
    yunet_ruta = os.path.join(base, "face_detection_yunet.onnx")
    if os.path.exists(yunet_ruta):
        try:
            _yunet = cv2.FaceDetectorYN.create(
                yunet_ruta, "", (640, 480),
                score_threshold=0.4, nms_threshold=0.3, top_k=5)
            print(f"[DET] YuNet: {yunet_ruta}")
        except Exception as e:
            print(f"[DET] YuNet error: {e}")
    else:
        print("[DET] YuNet no encontrado")

    # ── DNN SSD ───────────────────────────────────────────────────────────────
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

    # ── Haar frontal ──────────────────────────────────────────────────────────
    # Busca primero en models/, luego en el directorio de OpenCV
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
            print(f"[DET] Haar frontal: {ruta}")
            break

    if _haar_frontal is None:
        print("[DET] ERROR: Haar frontal no encontrado")

    # ── Haar perfil ───────────────────────────────────────────────────────────
    ruta_perfil = None
    try:
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            ruta_perfil = cv2.data.haarcascades + "haarcascade_profileface.xml"
    except:
        pass
    if ruta_perfil and os.path.exists(ruta_perfil):
        _haar_perfil = cv2.CascadeClassifier(ruta_perfil)
        print(f"[DET] Haar perfil: {ruta_perfil}")
    else:
        print("[DET] Haar perfil no encontrado (se usara frontal con umbral relajado)")


def _bbox_mayor(lista):
    """De una lista de (x,y,w,h,conf), retorna la de mayor area."""
    if not lista:
        return None
    return max(lista, key=lambda c: c[2] * c[3])


def _detectar_dnn(frame, tipo_esperado=None):
    """
    Detecta la cara principal segun el angulo esperado.

    Para pasos frontales:  YuNet -> DNN SSD -> Haar frontal
    Para pasos de perfil:  YuNet -> DNN SSD -> Haar perfil -> Haar frontal relajado
    """
    _init_detectores()
    h_img, w_img = frame.shape[:2]
    resultados   = []
    es_perfil = tipo_esperado in (TIPO_PERFIL_D, TIPO_PERFIL_I)

    # ── YuNet (maneja todos los angulos) ──────────────────────────────────────
    if _yunet is not None:
        # Ajustar umbrales para perfiles completos
        if es_perfil:
            _yunet.setScoreThreshold(0.3)
            _yunet.setTopK(10)
        else:
            _yunet.setScoreThreshold(0.4)
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
                if cf >= (0.3 if es_perfil else 0.4) and w > 20 and h > 20:
                    resultados.append((x, y, min(w_img-x, w), min(h_img-y, h), cf))
        if resultados:
            resultados.sort(key=lambda c: c[2]*c[3], reverse=True)
            # Guardar keypoints de la cara mas grande para clasificar angulo
            if faces is not None and len(faces) > 0:
                idx = int(np.argmax([f[2]*f[3] for f in faces]))
                _yunet_kpts = faces[idx][4:10].copy()
            return resultados
        _yunet_kpts = None

    # ── DNN SSD (maneja todos los angulos) ────────────────────────────────────
    if _dnn_net is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 117.0, 123.0))
        _dnn_net.setInput(blob)
        dets = _dnn_net.forward()
        umbral_conf = 0.35 if es_perfil else 0.45
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
            return resultados

    # ── Haar — estrategia segun angulo esperado ───────────────────────────────
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)

    if es_perfil and _haar_perfil is not None:
        # Para PERFIL_D (usuario gira a su derecha): en el frame ya volteado
        # se ve el perfil izquierdo de la imagen → usar cascade directo
        # Para PERFIL_I (usuario gira a su izquierda): se ve el perfil derecho
        # → voltear frame para que la cascade detecte
        if tipo_esperado == TIPO_PERFIL_I:
            gris_det = cv2.flip(gris, 1)
        else:
            gris_det = gris

        caras = _haar_perfil.detectMultiScale(
            gris_det, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))

        if len(caras) > 0:
            for (x, y, w, h) in caras:
                if tipo_esperado == TIPO_PERFIL_I:
                    x = w_img - x - w   # deshacer el flip de coordenadas
                resultados.append((int(x), int(y), int(w), int(h), 0.85))
            resultados.sort(key=lambda c: c[2]*c[3], reverse=True)
            return resultados

    # Haar frontal — para frontales usa parametros normales,
    # para perfiles usa parametros mas relajados para captar caras giradas
    if _haar_frontal is not None:
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
            resultados.sort(key=lambda c: c[2]*c[3], reverse=True)

    return resultados


# =============================================================================
#  CLASIFICACION DE ANGULO
# =============================================================================

_lm_det    = None
_buf_yaw   = []
_BUF_N_YAW = 3
_yunet_kpts = None   # keypoints YuNet [oj_dx,oj_dy,oj_ix,oj_iy,nz_x,nz_y]


def _get_lm_detector():
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
    print("[LBF] No disponible — usando Sobel")
    return None


def _calcular_yaw_lbf(frame_gris, bbox):
    """
    Yaw por asimetria de distancias ojo-nariz (LBF landmarks).
    Funciona correctamente con el frame ya volteado.
    """
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
        ojo_izq = lm[36].astype(np.float64)   # ojo izq en imagen = ojo der del usuario
        ojo_der = lm[45].astype(np.float64)   # ojo der en imagen = ojo izq del usuario
        d_izq   = float(np.linalg.norm(nariz - ojo_izq))
        d_der   = float(np.linalg.norm(nariz - ojo_der))
        total   = d_izq + d_der + 1e-6
        # Con frame volteado: usuario gira DERECHA → ojo_izq de imagen se aleja
        # d_izq > d_der → (d_der - d_izq) < 0 → negativo → necesitamos negarlo
        return -((d_der - d_izq) / total) * 120.0
    except Exception:
        return None


def _calcular_yaw_sobel(frame_gris, bbox, frame_shape):
    """
    Yaw por asimetria de bordes Sobel.
    El signo se niega porque el frame ya viene volteado desde interfaz.py.
    Sin el negativo: usuario gira DERECHA → izq del frame tiene mas bordes
    → (der-izq) negativo → clasificaria como IZQUIERDA (incorrecto).
    Con el negativo: se corrige el sentido.
    """
    x, y, w, h = bbox
    x1 = max(0, x);            y1 = max(0, y)
    x2 = min(frame_shape[1], x + w)
    y2 = min(frame_shape[0], y + h)
    recorte = frame_gris[y1:y2, x1:x2]
    if recorte.size == 0:
        return 0.0
    cara = cv2.resize(recorte, (128, 128))
    gx   = cv2.Sobel(cara, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(cara, cv2.CV_32F, 0, 1, ksize=3)
    b    = np.abs(gx) + np.abs(gy)
    w_   = b.shape[1]
    m    = int(w_ * 0.12)
    izq  = float(np.mean(b[:, :w_//2 - m]))
    der  = float(np.mean(b[:, w_//2 + m:]))
    # Negado para compensar el flip del frame
    return -((der - izq) / (der + izq + 1e-6)) * 120.0


def _clasificar_angulo(frame_gris, bbox, frame_shape, tipo_esperado=None):
    global _buf_yaw, _yunet_kpts

    yaw = None
    # 1. Keypoints YuNet: precisos, funcionan en perfiles completos
    if _yunet_kpts is not None:
        try:
            od = np.array([_yunet_kpts[0], _yunet_kpts[1]], dtype=np.float64)
            oi = np.array([_yunet_kpts[2], _yunet_kpts[3]], dtype=np.float64)
            nz = np.array([_yunet_kpts[4], _yunet_kpts[5]], dtype=np.float64)
            if np.linalg.norm(od)>1 and np.linalg.norm(oi)>1 and np.linalg.norm(nz)>1:
                d_der = float(np.linalg.norm(nz - od))
                d_izq = float(np.linalg.norm(nz - oi))
                # Frame ya flippeado: imagen-derecha = ojo izq del usuario
                # usuario gira DERECHA => d_der > d_izq => asim > 0 => PERFIL_DER
                asim  = (d_der - d_izq) / (d_der + d_izq + 1e-6)
                yaw   = asim * 120.0
        except Exception:
            yaw = None

    # 2. LBF 68 landmarks -- backup
    if yaw is None:
        yaw = _calcular_yaw_lbf(frame_gris, bbox)

    # 3. Sobel -- ultimo recurso
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
    """
    Extrae yaw y pitch usando LBF landmarks.
    Pitch no implementado, retorna None.
    """
    yaw = _calcular_yaw_lbf(gris, bbox)
    pitch = None
    return yaw, pitch


# =============================================================================
#  ZONAS LBP POR ANGULO  (8 zonas x 64 bins = 512 dims)
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
VECTOR_DIM = N_ZONAS * LBP_BINS   # 512


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
#  API PUBLICA
# =============================================================================

def preprocesar_cara(gris_zona):
    return cv2.GaussianBlur(_clahe.apply(gris_zona), (3, 3), 0)


def extraer_caracteristicas(frame, haar_path=None, modo="auto",
                             tipo_esperado=None):
    """
    Detecta cara y extrae vector LBP de 512 dims.
    Retorna (vector, coords, tipo) o (None, None, None).
    """
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
    denom = v1 + v2 + 1e-7
    return float(np.sum((v1 - v2) ** 2 / denom))


def dibujar_overlay(frame, coords, color, texto="", tipo=None):
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