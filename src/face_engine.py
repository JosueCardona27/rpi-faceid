"""
face_engine.py
==============
Motor de reconocimiento facial multiangulo con LBP.

Deteccion:  DNN SSD ResNet (principal, conf=0.18) + Haar frontal (backup)
Angulo:     LBF 68 landmarks -> asimetria nariz-ojos -> yaw suavizado
Vectores:   LBP uniforme (59 bins) por 7 zonas especificas de cada angulo

Angulos soportados:
  frontal    -- cara mirando directo a la camara
  perfil_der -- cara girada a su DERECHA (vemos su lado izquierdo en imagen)
  perfil_izq -- cara girada a su IZQUIERDA (vemos su lado derecho en imagen)

Principio de clasificacion por asimetria:
  Al girar la cara a su DERECHA:
    - El ojo derecho de la persona se aleja de la nariz en la imagen
    - El ojo izquierdo se acerca
    - d(ojo_der, nariz) > d(ojo_izq, nariz)  =>  yaw > 0  =>  PERFIL_DER

  Umbrales adaptativos:
    - Modo registro (tipo_esperado != None):  umbral = 8  (permisivo)
    - Modo acceso   (tipo_esperado = None):   umbral = 15 (estricto)
"""

import cv2
import numpy as np
import os

# ─── tipos de angulo ──────────────────────────────────────────────────────────
TIPO_FRONTAL  = "frontal"
TIPO_PERFIL_D = "perfil_der"
TIPO_PERFIL_I = "perfil_izq"
TIPO_ABAJO    = "abajo"     # por compatibilidad, no se clasifica activamente

_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))


# =============================================================================
#  DETECCION DE CARA  (DNN + Haar backup)
# =============================================================================

_dnn_net  = None
_haar_cas = None
_det_init = False


def _init_detectores():
    global _dnn_net, _haar_cas, _det_init
    if _det_init:
        return
    _det_init = True

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

    # DNN SSD ResNet -- deteccion principal
    proto = os.path.join(base, "opencv_face_detector.prototxt")
    caffe = os.path.join(base, "opencv_face_detector.caffemodel")
    if os.path.exists(proto) and os.path.exists(caffe):
        _dnn_net = cv2.dnn.readNetFromCaffe(proto, caffe)
        print(f"[DNN] Modelo cargado: {caffe}")
    else:
        print("[DNN] WARN: modelo no encontrado")

    # Haar frontal -- backup cuando DNN pierde perfiles
    haar_candidatos = [
        os.path.join(base, "haarcascade_frontalface_default.xml"),
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]
    try:
        haar_candidatos.append(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        pass

    for ruta in haar_candidatos:
        if os.path.exists(ruta):
            cas = cv2.CascadeClassifier(ruta)
            if not cas.empty():
                _haar_cas = cas
                print(f"[HAAR] Cascade backup: {ruta}")
                break


def _detectar_dnn(frame, conf_min=0.18):
    """
    Detecta caras en el frame con estrategia dual.

    1. DNN SSD con conf_min=0.18 (bajo para capturar perfiles parciales)
    2. Haar frontal muy permisivo si DNN no detecta nada

    Retorna lista de (x, y, w, h, conf) ordenada por area descendente.
    """
    _init_detectores()

    h_img, w_img = frame.shape[:2]
    resultados   = []

    # Intento 1: DNN
    if _dnn_net is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 117.0, 123.0))
        _dnn_net.setInput(blob)
        dets = _dnn_net.forward()

        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < conf_min:
                continue
            x1 = max(0, int(dets[0, 0, i, 3] * w_img))
            y1 = max(0, int(dets[0, 0, i, 4] * h_img))
            x2 = min(w_img, int(dets[0, 0, i, 5] * w_img))
            y2 = min(h_img, int(dets[0, 0, i, 6] * h_img))
            if x2 > x1 and y2 > y1:
                resultados.append((x1, y1, x2 - x1, y2 - y1, conf))

    if resultados:
        resultados.sort(key=lambda c: c[2] * c[3], reverse=True)
        return resultados

    # Intento 2: Haar muy permisivo
    if _haar_cas is not None and not _haar_cas.empty():
        gris  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris  = cv2.equalizeHist(gris)
        caras = _haar_cas.detectMultiScale(
            gris, scaleFactor=1.1, minNeighbors=2,
            minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(caras) > 0:
            resultados = [(int(x), int(y), int(w), int(h), 0.5)
                          for (x, y, w, h) in caras]
            resultados.sort(key=lambda c: c[2] * c[3], reverse=True)
            return resultados

    return []


# =============================================================================
#  CLASIFICACION DE ANGULO  (LBF landmarks + fallback Sobel)
# =============================================================================

_lm_det    = None
_buf_yaw   = []
_BUF_N_YAW = 5


def _get_lm_detector():
    """Carga el modelo LBF una sola vez."""
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
            print(f"[LBF] Modelo cargado: {ruta}")
            return _lm_det
        except Exception as e:
            print(f"[LBF] Error al cargar: {e}")
    _lm_det = False
    print("[LBF] No disponible -- usando asimetria de bordes como fallback")
    return None


def _calcular_yaw_lbf(frame_gris, bbox):
    """
    Calcula el yaw de la cabeza midiendo asimetria entre ojos y nariz.

    Indices LBF (de los 68 puntos):
      30 = punta de la nariz
      36 = esquina externa del ojo izquierdo
      45 = esquina externa del ojo derecho

    Retorna yaw en escala [-120, +120] aprox, o None si LBF falla.
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
        ojo_izq = lm[36].astype(np.float64)
        ojo_der = lm[45].astype(np.float64)

        d_izq = float(np.linalg.norm(nariz - ojo_izq))
        d_der = float(np.linalg.norm(nariz - ojo_der))
        total = d_izq + d_der + 1e-6

        # asim > 0 => cara a su DERECHA => TIPO_PERFIL_D
        # asim < 0 => cara a su IZQUIERDA => TIPO_PERFIL_I
        asim = (d_der - d_izq) / total

        return asim * 120.0

    except Exception:
        return None


def _calcular_yaw_sobel(frame_gris, bbox, frame_shape):
    """Fallback: yaw por asimetria de bordes Sobel."""
    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_shape[1], x + w)
    y2 = min(frame_shape[0], y + h)

    recorte = frame_gris[y1:y2, x1:x2]
    if recorte.size == 0:
        return 0.0

    cara = cv2.resize(recorte, (128, 128))
    gx   = cv2.Sobel(cara, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(cara, cv2.CV_32F, 0, 1, ksize=3)
    b    = np.abs(gx) + np.abs(gy)
    h_, w_ = b.shape
    m    = int(w_ * 0.12)
    izq  = float(np.mean(b[:, :w_ // 2 - m]))
    der  = float(np.mean(b[:, w_ // 2 + m:]))
    asim = (der - izq) / (der + izq + 1e-6)

    return asim * 120.0


def _clasificar_angulo(frame_gris, bbox, frame_shape, tipo_esperado=None):
    """
    Clasifica el angulo de la cabeza: frontal, perfil_der, o perfil_izq.

    tipo_esperado: angulo que el sistema espera en este momento.
      TIPO_PERFIL_D o TIPO_PERFIL_I => umbral permisivo (8) para registro
      None o TIPO_FRONTAL           => umbral normal (15) para acceso
    """
    global _buf_yaw

    yaw = _calcular_yaw_lbf(frame_gris, bbox)
    if yaw is None:
        yaw = _calcular_yaw_sobel(frame_gris, bbox, frame_shape)

    # Suavizar con buffer
    _buf_yaw.append(yaw)
    if len(_buf_yaw) > _BUF_N_YAW:
        _buf_yaw.pop(0)
    yaw_s = float(np.mean(_buf_yaw))

    # Umbral adaptativo
    if tipo_esperado in (TIPO_PERFIL_D, TIPO_PERFIL_I):
        umbral = 8.0    # registro guiado: usuario intentando mostrar perfil
    else:
        umbral = 15.0   # acceso libre: evitar falsos positivos

    if yaw_s > umbral:
        return TIPO_PERFIL_D
    elif yaw_s < -umbral:
        return TIPO_PERFIL_I
    else:
        return TIPO_FRONTAL


def _extraer_angulos_lbf(frame_gris, bbox, fw, fh):
    """Wrapper de compatibilidad para diagnostico.py."""
    yaw = _calcular_yaw_lbf(frame_gris, bbox)
    return yaw, 0.0


# =============================================================================
#  ZONAS LBP POR ANGULO  (sobre imagen 128x128)
# =============================================================================

ZONAS_FRONTAL = [
    (0,  40,  0,  128, "frente"),
    (28, 65,  0,   58, "ojo_izq"),
    (28, 65, 70,  128, "ojo_der"),
    (55, 92, 28,  100, "nariz"),
    (62, 100, 0,   50, "mejilla_izq"),
    (62, 100, 78, 128, "mejilla_der"),
    (88, 128, 14, 114, "boca_menton"),
]

ZONAS_PERFIL_D = [
    # Lado izquierdo visible cuando cara gira a su derecha
    (0,  40,  0,  128, "frente"),
    (20, 60,  0,   60, "ojo_izq"),
    (45, 85,  0,   65, "nariz_lat"),
    (55, 95,  0,   55, "mejilla_izq"),
    (65, 110, 0,   50, "mandibula"),
    (25, 75,  0,   40, "pomulo"),
    (82, 128, 0,   75, "menton"),
]

ZONAS_PERFIL_I = [
    # Lado derecho visible -- espejo de PERFIL_D
    (0,  40,  0,  128, "frente"),
    (20, 60, 68,  128, "ojo_der"),
    (45, 85, 63,  128, "nariz_lat"),
    (55, 95, 73,  128, "mejilla_der"),
    (65, 110, 78, 128, "mandibula"),
    (25, 75, 88,  128, "pomulo"),
    (82, 128, 53, 128, "menton"),
]

ZONAS_ABAJO = ZONAS_FRONTAL   # compatibilidad

ZONAS_POR_TIPO = {
    TIPO_FRONTAL:  ZONAS_FRONTAL,
    TIPO_PERFIL_D: ZONAS_PERFIL_D,
    TIPO_PERFIL_I: ZONAS_PERFIL_I,
    TIPO_ABAJO:    ZONAS_FRONTAL,
}

ZONAS      = ZONAS_FRONTAL
N_ZONAS    = len(ZONAS_FRONTAL)
LBP_BINS   = 59
VECTOR_DIM = N_ZONAS * LBP_BINS   # 413


# =============================================================================
#  CALCULO LBP
# =============================================================================

VAR_MIN  = 250.0
VAR_FULL = 800.0

_UNIFORM_MAP = None


def _build_uniform_map():
    umap = np.full(256, 58, dtype=np.int32)
    idx  = 0
    for code in range(256):
        b = format(code, '08b')
        t = sum(b[i] != b[(i + 1) % 8] for i in range(8))
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
        return np.zeros(59, dtype=np.float32), 0.0
    lbp_map  = _lbp_imagen(zona)
    umap     = _get_uniform_map()
    hist     = np.bincount(umap[lbp_map.flatten()],
                           minlength=59).astype(np.float32)
    total    = hist.sum()
    if total > 0:
        hist /= total
    varianza = float(np.var(zona.astype(np.float32)))
    return hist, varianza


def _varianza_a_peso(var):
    if var < VAR_MIN:
        return 0.0
    if var >= VAR_FULL:
        return 1.0
    return (var - VAR_MIN) / (VAR_FULL - VAR_MIN)


# =============================================================================
#  API PUBLICA
# =============================================================================

def preprocesar_cara(gris_zona):
    """CLAHE + GaussianBlur para normalizar iluminacion."""
    return cv2.GaussianBlur(_clahe.apply(gris_zona), (3, 3), 0)


def extraer_caracteristicas(frame, haar_path=None, modo="auto",
                             tipo_esperado=None):
    """
    Detecta cara y extrae vector LBP con zonas del angulo actual.

    tipo_esperado: si se pasa, usa umbral de yaw mas permisivo para ese angulo.

    Retorna: (vector, pesos, coords, tipo) o (None, None, None, None)
    """
    caras = _detectar_dnn(frame)
    if not caras:
        return None, None, None, None

    x, y, w, h, _ = caras[0]
    h_img, w_img  = frame.shape[:2]
    x1, y1        = max(0, x), max(0, y)
    x2, y2        = min(w_img, x + w), min(h_img, y + h)

    gris_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recorte   = preprocesar_cara(gris_full[y1:y2, x1:x2])
    if recorte.size == 0:
        return None, None, None, None

    cara128 = cv2.resize(recorte, (128, 128))

    tipo = _clasificar_angulo(
        gris_full, (x1, y1, x2 - x1, y2 - y1),
        frame.shape, tipo_esperado=tipo_esperado)

    zonas = ZONAS_POR_TIPO.get(tipo, ZONAS_FRONTAL)
    hists = []
    pesos = []
    for r0, r1, c0, c1, _ in zonas:
        hist, var = _histograma_zona(cara128, r0, r1, c0, c1)
        hists.append(hist)
        pesos.append(_varianza_a_peso(var))

    vector = np.concatenate(hists).astype(np.float32)
    pesos  = np.array(pesos, dtype=np.float32)

    return vector, pesos, (x1, y1, x2 - x1, y2 - y1), tipo


def distancia_ponderada(v1, p1, v2, p2):
    """
    Distancia chi-cuadrado ponderada entre dos vectores LBP.
    Solo usa zonas donde AMBOS tienen peso > 0.15.
    Retorna (distancia, n_zonas_usadas).
    """
    dist_total   = 0.0
    peso_total   = 0.0
    zonas_usadas = 0

    for z in range(N_ZONAS):
        w = min(p1[z], p2[z])
        if w < 0.15:
            continue
        i0    = z * LBP_BINS
        i1    = i0 + LBP_BINS
        h1    = v1[i0:i1]
        h2    = v2[i0:i1]
        denom = h1 + h2 + 1e-7
        chi2  = float(np.sum((h1 - h2) ** 2 / denom))
        dist_total   += w * chi2
        peso_total   += w
        zonas_usadas += 1

    if zonas_usadas < 2 or peso_total < 0.1:
        return float('inf'), 0

    return dist_total / peso_total, zonas_usadas


def dibujar_overlay(frame, coords, color, texto="", tipo=None):
    """Dibuja esquinas de deteccion con color por tipo y etiqueta."""
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
        ((x,     y),   (x + L,     y)),
        ((x,     y),   (x,         y + L)),
        ((x + w, y),   (x + w - L, y)),
        ((x + w, y),   (x + w,     y + L)),
        ((x,     y+h), (x + L,     y + h)),
        ((x,     y+h), (x,         y + h - L)),
        ((x + w, y+h), (x + w - L, y + h)),
        ((x + w, y+h), (x + w,     y + h - L)),
    ]:
        cv2.line(frame, p1, p2, c, 2)

    etiquetas = {
        TIPO_FRONTAL:  "FRONTAL",
        TIPO_PERFIL_D: "PERFIL DER",
        TIPO_PERFIL_I: "PERFIL IZQ",
        TIPO_ABAJO:    "ABAJO",
    }
    if tipo in etiquetas:
        cv2.putText(frame, etiquetas[tipo],
                    (x, y + h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    if texto:
        cv2.putText(frame, texto, (x, max(14, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame