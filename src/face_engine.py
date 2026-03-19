"""
face_engine.py - Version YuNet corregida
==========================================
CORRECCIONES:
  1. Perfiles invertidos: el frame llega con cv2.flip(raw,1) desde interfaz.
     YuNet devuelve landmarks sobre el frame ya volteado, por lo que la
     asimetria izq/der queda correcta SIN compensacion adicional.
     Se elimino la inversion erronea de TIPO_PERFIL_I <-> TIPO_PERFIL_D.
  2. Umbral asimetria ajustado: 0.28 → 0.22 para detectar perfiles antes.
  3. Fallback Sobel con umbral ±40 (no ±25) para no generar falsos perfiles.

REQUISITOS:
  - OpenCV >= 4.5.4  (tienes 4.13 ✅)
  - models/face_detection_yunet_2023mar.onnx  ✅
"""

import os
import cv2
import numpy as np

TIPO_FRONTAL  = "frontal"
TIPO_PERFIL_D = "perfil_der"
TIPO_PERFIL_I = "perfil_izq"

_clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS   = os.path.join(_BASE_DIR, "..", "models")


# =============================================================================
#  DETECTOR YUNET  (singleton)
# =============================================================================

_yunet        = None
_haar_frontal = None
_haar_perfil  = None
_det_init     = False


def _init_detectores():
    global _yunet, _haar_frontal, _haar_perfil, _det_init
    if _det_init:
        return
    _det_init = True

    # ── YuNet ─────────────────────────────────────────────────────────────────
    for nombre in ("face_detection_yunet_2023mar.onnx", "face_detection_yunet.onnx"):
        ruta = os.path.join(_MODELS, nombre)
        if os.path.exists(ruta):
            try:
                _yunet = cv2.FaceDetectorYN.create(
                    ruta, "", (640, 480),
                    score_threshold=0.55,
                    nms_threshold=0.3,
                    top_k=5000
                )
                print(f"[DET] ✅ YuNet cargado: {ruta}")
                break
            except Exception as e:
                print(f"[DET] ⚠️  YuNet error: {e}")

    if _yunet is None:
        print(f"[DET] ⚠️  YuNet NO encontrado en {_MODELS}")

    # ── Haar frontal (fallback) ───────────────────────────────────────────────
    for ruta in [
        os.path.join(_MODELS, "haarcascade_frontalface_default.xml"),
        os.path.join(_BASE_DIR, "haarcascade_frontalface_default.xml"),
        *(([cv2.data.haarcascades + "haarcascade_frontalface_default.xml"])
          if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") else [])
    ]:
        if os.path.exists(ruta):
            c = cv2.CascadeClassifier(ruta)
            if not c.empty():
                _haar_frontal = c
                print(f"[DET] Haar frontal (fallback): {ruta}")
                break

    # ── Haar perfil (fallback) ────────────────────────────────────────────────
    for ruta in [
        os.path.join(_MODELS, "haarcascade_profileface.xml"),
        os.path.join(_BASE_DIR, "haarcascade_profileface.xml"),
        *(([cv2.data.haarcascades + "haarcascade_profileface.xml"])
          if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") else [])
    ]:
        if os.path.exists(ruta):
            c = cv2.CascadeClassifier(ruta)
            if not c.empty():
                _haar_perfil = c
                print(f"[DET] Haar perfil (fallback): {ruta}")
                break

    if _yunet is None and _haar_frontal is None:
        print("[DET] ❌ Ningun detector disponible.")


# =============================================================================
#  DETECCION
# =============================================================================

def _detectar_caras_yunet(frame):
    """
    YuNet detecta sobre el frame tal como llega (ya volteado por interfaz.py).
    Retorna lista de (x, y, w, h, score, kps_dict).
    """
    h_img, w_img = frame.shape[:2]
    _yunet.setInputSize((w_img, h_img))
    _, faces = _yunet.detect(frame)

    detecciones = []
    if faces is None:
        return detecciones

    for face in faces:
        x = int(face[0]); y = int(face[1])
        w = int(face[2]); h = int(face[3])
        score = float(face[14])

        x = max(0, x); y = max(0, y)
        w = min(w, w_img - x); h = min(h, h_img - y)
        if w < 15 or h < 15:
            continue

        kps = {
            "ojo_der":  (float(face[4]),  float(face[5])),
            "ojo_izq":  (float(face[6]),  float(face[7])),
            "nariz":    (float(face[8]),  float(face[9])),
            "boca_izq": (float(face[10]), float(face[11])),
            "boca_der": (float(face[12]), float(face[13])),
        }
        detecciones.append((x, y, w, h, round(score, 3), kps))

    detecciones.sort(key=lambda d: d[2] * d[3], reverse=True)
    return detecciones


def _detectar_caras_haar(frame, tipo_esperado=None):
    h_img, w_img = frame.shape[:2]
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = _clahe.apply(gris)
    res  = []

    if tipo_esperado != TIPO_PERFIL_D and tipo_esperado != TIPO_PERFIL_I:
        if _haar_frontal is not None:
            caras = _haar_frontal.detectMultiScale(
                gris, scaleFactor=1.03, minNeighbors=3, minSize=(40, 40))
            for (x, y, w, h) in caras:
                res.append((int(x), int(y), int(w), int(h), 0.75, {}))

    if tipo_esperado in (TIPO_PERFIL_D, TIPO_PERFIL_I) or tipo_esperado is None:
        if _haar_perfil is not None:
            # Para perfil izquierdo en frame ya volteado, buscar sin flip extra
            caras = _haar_perfil.detectMultiScale(
                gris, scaleFactor=1.02, minNeighbors=2, minSize=(30, 30))
            for (x, y, w, h) in caras:
                res.append((int(x), int(y), int(w), int(h), 0.70, {}))

            # Buscar tambien perfil espejado
            gris_flip = cv2.flip(gris, 1)
            caras = _haar_perfil.detectMultiScale(
                gris_flip, scaleFactor=1.02, minNeighbors=2, minSize=(30, 30))
            for (x, y, w, h) in caras:
                x_real = w_img - x - w
                res.append((int(x_real), int(y), int(w), int(h), 0.68, {}))

    res.sort(key=lambda d: d[2] * d[3], reverse=True)
    return res


def _detectar_caras(frame, tipo_esperado=None):
    _init_detectores()
    if _yunet is not None:
        dets = _detectar_caras_yunet(frame)
    else:
        dets = _detectar_caras_haar(frame, tipo_esperado)
    return [(d[0], d[1], d[2], d[3], d[4]) for d in dets]


# =============================================================================
#  CLASIFICACION DE ANGULO
# =============================================================================

_buf_yaw   = []
_BUF_N_YAW = 8


def _clasificar_angulo_yunet(frame):
    """
    Clasifica el angulo usando landmarks de YuNet sobre el frame ya volteado.

    NOTA IMPORTANTE sobre el flip:
    - interfaz.py hace cv2.flip(raw, 1) antes de pasar el frame
    - Por tanto el frame es un espejo: lo que el usuario ve como "su derecha"
      aparece en el lado DERECHO de la imagen (como un espejo)
    - YuNet en este frame: ojo_der del usuario → lado derecho de imagen
    - La asimetria se calcula directamente sin inversion adicional

    Logica:
      - Si ojo_der esta MAS lejos del centro que ojo_izq → usuario girado a izq → PERFIL_IZQ
      - Si ojo_izq esta MAS lejos del centro que ojo_der → usuario girado a der → PERFIL_DER
    """
    if _yunet is None:
        return None

    h_img, w_img = frame.shape[:2]
    _yunet.setInputSize((w_img, h_img))
    _, faces = _yunet.detect(frame)

    if faces is None or len(faces) == 0:
        return None

    face    = faces[0]
    x       = float(face[0])
    w       = float(face[2])
    cx_cara = x + w / 2.0

    x_ojo_der = float(face[4])   # ojo derecho del usuario (lado der en imagen espejo)
    x_ojo_izq = float(face[6])   # ojo izquierdo del usuario (lado izq en imagen espejo)
    x_nariz   = float(face[8])

    # Distancia de cada ojo al centro de la cara
    dist_der = cx_cara - x_ojo_der   # + → ojo der esta a la izq del centro (perfil izq)
    dist_izq = x_ojo_izq - cx_cara   # + → ojo izq esta a la der del centro (perfil der)

    total     = abs(dist_der) + abs(dist_izq) + 1e-6
    asimetria = (dist_der - dist_izq) / total
    # asimetria > 0 → ojo_der mas cerca del centro = cara girada a la izquierda

    _buf_yaw.append(asimetria)
    if len(_buf_yaw) > _BUF_N_YAW:
        _buf_yaw.pop(0)
    asimetria_suave = float(np.median(_buf_yaw))

    # Umbral 0.22: mas sensible que antes (0.28) para capturar perfiles reales
    if asimetria_suave > 0.22:
        return TIPO_PERFIL_I
    elif asimetria_suave < -0.22:
        return TIPO_PERFIL_D

    # Confirmacion con nariz
    desv_nariz = (x_nariz - cx_cara) / (w + 1e-6)
    if desv_nariz > 0.10:
        return TIPO_PERFIL_D
    elif desv_nariz < -0.10:
        return TIPO_PERFIL_I

    return TIPO_FRONTAL


def _calcular_yaw_sobel(frame_gris, bbox):
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_gris.shape[1], x + w), min(frame_gris.shape[0], y + h)
    recorte = frame_gris[y1:y2, x1:x2]
    if recorte.size == 0:
        return 0.0
    try:
        cara      = cv2.resize(recorte, (128, 128))
        gx        = cv2.Sobel(cara, cv2.CV_32F, 1, 0, ksize=7)
        gx_abs    = np.abs(gx)
        w_col     = cara.shape[1] // 3
        izq       = np.mean(gx_abs[:, :w_col])
        der       = np.mean(gx_abs[:, 2 * w_col:])
        asimetria = (der - izq) / (der + izq + 1e-6)
        return asimetria * 100.0
    except Exception as e:
        print(f"[Sobel] Error: {e}")
        return 0.0


def _clasificar_angulo(frame, bbox, frame_shape, tipo_esperado=None):
    """
    Prioridad:
      1. tipo_esperado (registro guiado)
      2. YuNet landmarks
      3. Sobel fallback (umbral ±40, no ±25)
    """
    global _buf_yaw

    if tipo_esperado in (TIPO_FRONTAL, TIPO_PERFIL_D, TIPO_PERFIL_I):
        return tipo_esperado

    tipo_yn = _clasificar_angulo_yunet(frame)
    if tipo_yn is not None:
        return tipo_yn

    # Fallback Sobel
    try:
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yaw        = _calcular_yaw_sobel(frame_gris, bbox)
    except Exception:
        yaw = 0.0

    _buf_yaw.append(yaw)
    if len(_buf_yaw) > _BUF_N_YAW:
        _buf_yaw.pop(0)
    yaw_s = float(np.median(_buf_yaw)) if _buf_yaw else yaw

    # Umbral corregido: ±40 en lugar de ±25
    if yaw_s > 40.0:
        return TIPO_PERFIL_D
    elif yaw_s < -40.0:
        return TIPO_PERFIL_I
    else:
        return TIPO_FRONTAL


def _extraer_angulos_lbf(gris, bbox, fw, fh):
    return None, None


# =============================================================================
#  ZONAS LBP
# =============================================================================

ZONAS_FRONTAL = [
    (0,  40,  0,   128, "frente"),
    (28, 65,  0,   58,  "ojo_izq"),
    (28, 65,  70,  128, "ojo_der"),
    (55, 92,  28,  100, "nariz"),
    (62, 100, 0,   50,  "mejilla_izq"),
    (62, 100, 78,  128, "mejilla_der"),
    (88, 128, 14,  114, "boca_menton"),
    (20, 100, 0,   128, "cara_media"),
]

ZONAS_PERFIL_D = [
    (0,  40,  0,  128, "frente"),
    (20, 60,  0,  60,  "ojo"),
    (45, 85,  0,  65,  "nariz_lat"),
    (55, 95,  0,  55,  "mejilla"),
    (65, 110, 0,  50,  "mandibula"),
    (25, 75,  0,  40,  "pomulo"),
    (82, 128, 0,  75,  "menton"),
    (15, 105, 0,  80,  "perfil_media"),
]

ZONAS_PERFIL_I = [
    (0,  40,  0,   128, "frente"),
    (20, 60,  68,  128, "ojo"),
    (45, 85,  63,  128, "nariz_lat"),
    (55, 95,  73,  128, "mejilla"),
    (65, 110, 78,  128, "mandibula"),
    (25, 75,  88,  128, "pomulo"),
    (82, 128, 53,  128, "menton"),
    (15, 105, 48,  128, "perfil_media"),
]

ZONAS_POR_TIPO = {
    TIPO_FRONTAL:  ZONAS_FRONTAL,
    TIPO_PERFIL_D: ZONAS_PERFIL_D,
    TIPO_PERFIL_I: ZONAS_PERFIL_I,
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
        b     = format(code, "08b")
        trans = sum(b[i] != b[(i + 1) % 8] for i in range(8))
        if trans <= 2:
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
    pad    = np.pad(img, 1, mode="edge")
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
    hist59  = np.bincount(umap[lbp_map.flatten()], minlength=59).astype(np.float32)
    total   = hist59.sum()
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


def extraer_caracteristicas(frame, haar_path=None, modo="auto", tipo_esperado=None):
    caras = _detectar_caras(frame, tipo_esperado=tipo_esperado)
    if not caras:
        return None, None, None

    x, y, w, h, _ = caras[0]
    h_img, w_img   = frame.shape[:2]
    x1, y1 = max(0, x),         max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    gris_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recorte   = gris_full[y1:y2, x1:x2]
    if recorte.size == 0:
        return None, None, None

    recorte = preprocesar_cara(recorte)
    cara128 = cv2.resize(recorte, (128, 128))

    tipo = _clasificar_angulo(
        frame, (x1, y1, x2-x1, y2-y1), frame.shape,
        tipo_esperado=tipo_esperado)

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
        TIPO_FRONTAL:  (0, 212, 255),
        TIPO_PERFIL_D: (255, 165, 0),
        TIPO_PERFIL_I: (0, 165, 255),
    }
    c = colores_tipo.get(tipo, color)
    for p1, p2 in [
        ((x,y),(x+L,y)),((x,y),(x,y+L)),
        ((x+w,y),(x+w-L,y)),((x+w,y),(x+w,y+L)),
        ((x,y+h),(x+L,y+h)),((x,y+h),(x,y+h-L)),
        ((x+w,y+h),(x+w-L,y+h)),((x+w,y+h),(x+w,y+h-L)),
    ]:
        cv2.line(frame, p1, p2, c, 2)
    etiquetas = {TIPO_FRONTAL:"FRONTAL", TIPO_PERFIL_D:"PERFIL DER", TIPO_PERFIL_I:"PERFIL IZQ"}
    if tipo in etiquetas:
        cv2.putText(frame, etiquetas[tipo], (x, y+h+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    if texto:
        cv2.putText(frame, texto, (x, max(14, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def guardar_rostro_recortado(frame, nombre="persona", carpeta_base="dataset", tipo_esperado=None):
    from datetime import datetime
    vector, bbox, tipo = extraer_caracteristicas(frame, tipo_esperado=tipo_esperado)
    if bbox is None:
        return None, None, None, None
    x, y, w, h   = bbox
    h_img, w_img = frame.shape[:2]
    x1, y1 = max(0,x), max(0,y)
    x2, y2 = min(w_img,x+w), min(h_img,y+h)
    rostro = frame[y1:y2, x1:x2]
    if rostro.size == 0:
        return None, None, None, None
    ruta_dir = os.path.join(carpeta_base, nombre)
    os.makedirs(ruta_dir, exist_ok=True)
    marca   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archivo = f"{nombre}_{tipo}_{marca}.png"
    ruta    = os.path.join(ruta_dir, archivo)
    cv2.imwrite(ruta, rostro)
    return ruta, vector, bbox, tipo