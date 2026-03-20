"""
face_engine.py - Version YuNet corregida v4
=============================================
CORRECCIONES vs v3:

  BUG 5 — Falsos positivos en fondo sin rostro
           score_threshold=0.30 es necesario para OV5647, NO se cambia.
           El problema real: YuNet reporta detecciones cuyas landmarks
           tienen posiciones geometricamente imposibles para una cara
           humana (ojos debajo de la boca, nariz fuera del bbox, etc.).
           FIX: funcion _validar_landmarks() que descarta detecciones
                espurias DESPUES del threshold usando restricciones
                geometricas de caras humanas reales:
                - tamaño minimo 60x60 px (era 15x15, demasiado permisivo)
                - proporcion ancho/alto entre 0.45 y 1.85
                - ojos en el tercio superior del bbox
                - nariz entre ojos y boca verticalmente
                - boca debajo de los ojos con separacion minima
                - distancia interocular 20-72% del ancho del bbox
                - todos los landmarks dentro del bbox (+20% margen)

  BUG 6 — Angulo ignorado durante el registro guiado
           _clasificar_angulo devolvía tipo_esperado sin medir el
           angulo real → cualquier cara pasaba sin importar la pose.
           FIX: eliminar cortocircuito. extraer_caracteristicas pasa
                tipo_esperado=None a _clasificar_angulo. La comparacion
                angulo_real vs angulo_esperado la hace interfaz.py.
"""

import os
import cv2
import numpy as np

# ── Tipos de angulo ───────────────────────────────────────────────────────────
TIPO_FRONTAL  = "frontal"
TIPO_PERFIL_D = "perfil_der"
TIPO_PERFIL_I = "perfil_izq"

# ── CLAHE ─────────────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# ── Rutas base ────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS   = os.path.join(_BASE_DIR, "..", "models")


# =============================================================================
#  DETECTOR  (singleton)
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
        yunet_path = os.path.join(_MODELS, nombre)
        if os.path.exists(yunet_path):
            try:
                _yunet = cv2.FaceDetectorYN.create(
                    yunet_path, "", (640, 480),
                    score_threshold=0.30,   # NO se cambia — OV5647 lo necesita bajo
                    nms_threshold=0.3,
                    top_k=5000
                )
                print(f"[DET] ✅ YuNet cargado: {yunet_path}")
            except Exception as e:
                print(f"[DET] ⚠️  YuNet error: {e}")
                _yunet = None
            break

    if _yunet is None:
        print(f"[DET] ⚠️  YuNet NO encontrado en {_MODELS}")

    # ── Haar frontal ──────────────────────────────────────────────────────────
    rutas_f = [
        os.path.join(_MODELS, "haarcascade_frontalface_default.xml"),
        os.path.join(_BASE_DIR, "haarcascade_frontalface_default.xml"),
    ]
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        rutas_f.append(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    for ruta in rutas_f:
        if os.path.exists(ruta):
            clf = cv2.CascadeClassifier(ruta)
            if not clf.empty():
                _haar_frontal = clf
                print(f"[DET] Haar frontal (fallback): {ruta}")
                break

    # ── Haar perfil ───────────────────────────────────────────────────────────
    rutas_p = [
        os.path.join(_MODELS, "haarcascade_profileface.xml"),
        os.path.join(_BASE_DIR, "haarcascade_profileface.xml"),
    ]
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        rutas_p.append(cv2.data.haarcascades + "haarcascade_profileface.xml")
    for ruta in rutas_p:
        if os.path.exists(ruta):
            clf = cv2.CascadeClassifier(ruta)
            if not clf.empty():
                _haar_perfil = clf
                print(f"[DET] Haar perfil (fallback): {ruta}")
                break

    if _yunet is None and _haar_frontal is None:
        print("[DET] ❌ Ningun detector disponible.")


# =============================================================================
#  CACHE DE ULTIMA DETECCION YUNET
# =============================================================================

_ultimo_face_yunet = None   # array de 15 valores de YuNet


# =============================================================================
#  VALIDACION GEOMETRICA DE LANDMARKS  (BUG5 FIX)
# =============================================================================

def _validar_landmarks(face_row):
    """
    Valida que los landmarks de YuNet sean geometricamente consistentes
    con una cara humana real.

    Rechaza detecciones espurias producidas por ruido de camara,
    texturas, o fondos con colores uniformes (como el fondo morado
    de OV5647 con poca iluminacion). El score_threshold=0.30 se
    mantiene bajo para no perder caras reales, pero este filtro
    geometrico actua como segunda barrera de calidad.

    face_row layout (15 valores de YuNet):
      [0,1,2,3]   = x, y, w, h  del bounding box
      [4,5]       = ojo derecho x, y  (en imagen con flip horizontal)
      [6,7]       = ojo izquierdo x, y
      [8,9]       = nariz x, y
      [10,11]     = boca izquierda x, y
      [12,13]     = boca derecha x, y
      [14]        = score de confianza

    Retorna True si la deteccion parece una cara real.
    """
    fx = float(face_row[0])
    fy = float(face_row[1])
    fw = float(face_row[2])
    fh = float(face_row[3])

    # ── 1. Tamaño minimo ─────────────────────────────────────────────────────
    # 15x15 era demasiado permisivo. A 640x480, una cara real a
    # distancia util tiene al menos 60x60 px.
    if fw < 60 or fh < 60:
        return False

    # ── 2. Proporcion ancho/alto de la cara ──────────────────────────────────
    # Una cara humana vista de frente o perfil tiene ratio ~0.5-1.8.
    # Valores fuera de ese rango son ruido o artefactos.
    ratio = fw / (fh + 1e-6)
    if ratio < 0.45 or ratio > 1.85:
        return False

    # ── 3. Extraer landmarks ─────────────────────────────────────────────────
    x_od, y_od = float(face_row[4]),  float(face_row[5])   # ojo derecho (imagen)
    x_oi, y_oi = float(face_row[6]),  float(face_row[7])   # ojo izquierdo (imagen)
    x_n,  y_n  = float(face_row[8]),  float(face_row[9])   # nariz
    x_bl, y_bl = float(face_row[10]), float(face_row[11])  # boca izquierda
    x_br, y_br = float(face_row[12]), float(face_row[13])  # boca derecha

    y_ojos = (y_od + y_oi) / 2.0
    y_boca = (y_bl + y_br) / 2.0

    # ── 4. Ojos en el tercio superior del bbox ───────────────────────────────
    # En cualquier pose de cara humana los ojos estan en la mitad superior.
    if y_ojos > fy + fh * 0.65:
        return False

    # ── 5. Nariz entre ojos y boca (verticalmente) ──────────────────────────
    # Margen del 5% del alto para tolerar inclinaciones.
    margen_v = fh * 0.05
    if y_n < y_ojos - margen_v:
        return False
    if y_n > y_boca + margen_v:
        return False

    # ── 6. Boca debajo de los ojos con separacion minima ─────────────────────
    if y_boca <= y_ojos + fh * 0.10:
        return False

    # ── 7. Distancia interocular razonable ──────────────────────────────────
    # Para cara frontal: ~40-60% del ancho.
    # Para perfiles: puede bajar hasta ~20% (solo se ve un ojo).
    dist_ojos = abs(x_oi - x_od)
    if dist_ojos < fw * 0.20 or dist_ojos > fw * 0.72:
        return False

    # ── 8. Todos los landmarks dentro del bbox (margen 20%) ─────────────────
    # Un falso positivo tipico tiene landmarks muy fuera del bbox detecto.
    mx = fw * 0.20
    my = fh * 0.20
    for lx, ly in [(x_od, y_od), (x_oi, y_oi),
                   (x_n,  y_n),
                   (x_bl, y_bl), (x_br, y_br)]:
        if lx < fx - mx or lx > fx + fw + mx:
            return False
        if ly < fy - my or ly > fy + fh + my:
            return False

    return True


# =============================================================================
#  DETECCION
# =============================================================================

def _detectar_caras_yunet(frame):
    """
    Detecta caras con YuNet y aplica validacion geometrica de landmarks.

    BUG5 FIX: cada deteccion pasa por _validar_landmarks() antes de
    ser aceptada. El fondo morado/ruidoso de OV5647 produce detecciones
    con landmarks en posiciones imposibles para una cara real, que
    ahora son descartadas aqui.

    Retorna lista de (x, y, w, h, score).
    """
    global _ultimo_face_yunet

    h_img, w_img       = frame.shape[:2]
    _yunet.setInputSize((w_img, h_img))
    _, faces            = _yunet.detect(frame)
    _ultimo_face_yunet  = None

    if faces is None or len(faces) == 0:
        return []

    detecciones = []
    for face in faces:
        x = int(face[0]); y = int(face[1])
        w = int(face[2]); h = int(face[3])
        score = float(face[14])

        x = max(0, x);  y = max(0, y)
        w = min(w, w_img - x);  h = min(h, h_img - y)

        # BUG5 FIX: rechazar detecciones geometricamente invalidas
        if not _validar_landmarks(face):
            continue

        detecciones.append((x, y, w, h, round(score, 3), face))

    if not detecciones:
        return []

    # Ordenar por area — cara mas grande primero
    detecciones.sort(key=lambda d: d[2] * d[3], reverse=True)

    # Guardar landmarks de la cara principal para clasificacion de angulo
    _ultimo_face_yunet = detecciones[0][5]

    return [(d[0], d[1], d[2], d[3], d[4]) for d in detecciones]


def _detectar_caras_haar(frame, tipo_esperado=None):
    """Fallback Haar — cuando YuNet no detecta nada valido."""
    global _ultimo_face_yunet
    _ultimo_face_yunet = None

    h_img, w_img = frame.shape[:2]
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = _clahe.apply(gris)
    resultados = []

    # Frontal
    if tipo_esperado in (TIPO_FRONTAL, None):
        if _haar_frontal is not None:
            caras = _haar_frontal.detectMultiScale(
                gris, scaleFactor=1.03, minNeighbors=3, minSize=(40, 40)
            )
            for (x, y, w, h) in caras:
                resultados.append((int(x), int(y), int(w), int(h), 0.75))

    # Perfil derecho
    if tipo_esperado in (TIPO_PERFIL_D, None):
        if _haar_perfil is not None:
            caras = _haar_perfil.detectMultiScale(
                gris, scaleFactor=1.02, minNeighbors=2, minSize=(30, 30)
            )
            for (x, y, w, h) in caras:
                resultados.append((int(x), int(y), int(w), int(h), 0.70))

    # Perfil izquierdo (imagen invertida)
    if tipo_esperado in (TIPO_PERFIL_I, None):
        if _haar_perfil is not None:
            gris_flip = cv2.flip(gris, 1)
            caras = _haar_perfil.detectMultiScale(
                gris_flip, scaleFactor=1.02, minNeighbors=2, minSize=(30, 30)
            )
            for (x, y, w, h) in caras:
                x = w_img - x - w
                resultados.append((int(x), int(y), int(w), int(h), 0.70))

    resultados.sort(key=lambda d: d[2] * d[3], reverse=True)
    return resultados


def _detectar_caras(frame, tipo_esperado=None):
    """
    Interfaz unificada.
    Intenta YuNet primero (con validacion geometrica de landmarks).
    Si no detecta nada valido, usa Haar como fallback.
    Devuelve lista de (x, y, w, h, score).
    """
    _init_detectores()

    if _yunet is not None:
        dets = _detectar_caras_yunet(frame)
        if dets:
            return dets
        return _detectar_caras_haar(frame, tipo_esperado)

    return _detectar_caras_haar(frame, tipo_esperado)


# =============================================================================
#  CLASIFICACION DE ANGULO
# =============================================================================

_buf_yaw   = []
_BUF_N_YAW = 8


def _clasificar_angulo_con_landmarks(face_row):
    """
    Clasifica el angulo usando landmarks de YuNet ya detectados.

    face_row indices:
      [0,1,2,3]   = x, y, w, h del bbox
      [4,5]       = ojo derecho x,y  (derecho en imagen = izquierdo del usuario por flip)
      [6,7]       = ojo izquierdo x,y
      [8,9]       = nariz x,y
      [10,11]     = boca izquierda x,y
      [12,13]     = boca derecha x,y
      [14]        = score

    Inversion por flip horizontal:
      El frame viene con cv2.flip(raw,1). Cuando el usuario gira a su DERECHA,
      en la imagen aparece girando a su IZQUIERDA. Por eso se intercambia
      PERFIL_I <-> PERFIL_D respecto a lo que indica la asimetria de imagen.
    """
    x     = float(face_row[0])
    w     = float(face_row[2])
    cx    = x + w / 2.0

    x_od  = float(face_row[4])   # ojo der en imagen
    x_oi  = float(face_row[6])   # ojo izq en imagen
    x_n   = float(face_row[8])   # nariz

    # ── Asimetria de ojos ─────────────────────────────────────────────────────
    dist_od   = cx - x_od
    dist_oi   = x_oi - cx
    total     = abs(dist_od) + abs(dist_oi) + 1e-6
    asimetria = (dist_od - dist_oi) / total

    _buf_yaw.append(asimetria)
    if len(_buf_yaw) > _BUF_N_YAW:
        _buf_yaw.pop(0)
    asm = float(np.median(_buf_yaw))

    # Intercambiado PERFIL_I <-> PERFIL_D por flip horizontal
    if asm > 0.40:
        return TIPO_PERFIL_D
    elif asm < -0.40:
        return TIPO_PERFIL_I

    # ── Posicion de nariz (confirmacion) ─────────────────────────────────────
    desv = (x_n - cx) / (w + 1e-6)
    if desv > 0.15:
        return TIPO_PERFIL_I   # intercambiado por flip
    elif desv < -0.15:
        return TIPO_PERFIL_D   # intercambiado por flip

    return TIPO_FRONTAL


def _calcular_yaw_sobel(frame_gris, bbox):
    """Fallback Sobel — solo cuando no hay landmarks YuNet."""
    x, y, w, h = bbox
    x1, y1     = max(0, x), max(0, y)
    x2, y2     = min(frame_gris.shape[1], x + w), min(frame_gris.shape[0], y + h)
    recorte    = frame_gris[y1:y2, x1:x2]

    if recorte.size == 0:
        return 0.0
    try:
        cara  = cv2.resize(recorte, (128, 128))
        gx    = cv2.Sobel(cara, cv2.CV_32F, 1, 0, ksize=7)
        gabs  = np.abs(gx)
        wc    = cara.shape[1] // 3
        izq   = np.mean(gabs[:, :wc])
        der   = np.mean(gabs[:, 2*wc:])
        return (der - izq) / (der + izq + 1e-6) * 100.0
    except Exception as e:
        print(f"[Sobel] Error: {e}")
        return 0.0


def _clasificar_angulo(frame, bbox, frame_shape, tipo_esperado=None):
    """
    Clasifica el angulo REAL de la cara detectada.

    BUG6 FIX: se elimino el bloque que devolvía tipo_esperado sin
    medir el angulo real. Ahora siempre se mide usando landmarks
    (YuNet, si hay cache) o Sobel (fallback Haar).

    tipo_esperado se ignora aqui. La comparacion angulo_real vs
    angulo_esperado la hace el llamador (interfaz._capturar_registro).

    Prioridad:
      1. YuNet landmarks del cache → precision con geometria real
      2. Sobel fallback → cuando YuNet no detecto nada valido
    """
    global _buf_yaw

    # YuNet: usar cache de la deteccion ya hecha (ya paso validacion)
    if _ultimo_face_yunet is not None:
        return _clasificar_angulo_con_landmarks(_ultimo_face_yunet)

    # Fallback Sobel (Haar no da landmarks)
    try:
        fg  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yaw = _calcular_yaw_sobel(fg, bbox)
    except Exception:
        yaw = 0.0

    _buf_yaw.append(yaw)
    if len(_buf_yaw) > _BUF_N_YAW:
        _buf_yaw.pop(0)
    ys = float(np.median(_buf_yaw)) if _buf_yaw else yaw

    if ys > 40.0:
        return TIPO_PERFIL_D
    elif ys < -40.0:
        return TIPO_PERFIL_I
    else:
        return TIPO_FRONTAL


def _extraer_angulos_lbf(gris, bbox, fw, fh):
    """Compatibilidad con diagnostico.py — siempre retorna None."""
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
    hist59  = np.bincount(
        umap[lbp_map.flatten()], minlength=59
    ).astype(np.float32)

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


def extraer_caracteristicas(frame, haar_path=None, modo="auto", tipo_esperado=None):
    """
    Detecta cara (con validacion geometrica), clasifica angulo REAL
    y extrae vector LBP.

    BUG5 FIX: _detectar_caras ya aplica _validar_landmarks(), asi que
    aqui solo llegan detecciones geometricamente validas.

    BUG6 FIX: tipo_esperado se pasa al detector Haar (para orientar
    el cascade), pero _clasificar_angulo recibe tipo_esperado=None
    para que siempre mida el angulo real de la camara.
    """
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

    # BUG6 FIX: tipo_esperado=None — siempre clasifica el angulo real
    tipo = _clasificar_angulo(
        frame,
        (x1, y1, x2 - x1, y2 - y1),
        frame.shape,
        tipo_esperado=None
    )

    zonas  = ZONAS_POR_TIPO.get(tipo, ZONAS_FRONTAL)
    vector = np.concatenate([
        _histograma_zona(cara128, r0, r1, c0, c1)
        for r0, r1, c0, c1, _ in zonas
    ]).astype(np.float32)

    return vector, (x1, y1, x2 - x1, y2 - y1), tipo


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
        ((x, y),         (x + L, y)),           ((x, y),         (x, y + L)),
        ((x + w, y),     (x + w - L, y)),       ((x + w, y),     (x + w, y + L)),
        ((x, y + h),     (x + L, y + h)),       ((x, y + h),     (x, y + h - L)),
        ((x + w, y + h), (x + w - L, y + h)),   ((x + w, y + h), (x + w, y + h - L)),
    ]:
        cv2.line(frame, p1, p2, c, 2)

    etiquetas = {
        TIPO_FRONTAL:  "FRONTAL",
        TIPO_PERFIL_D: "PERFIL DER",
        TIPO_PERFIL_I: "PERFIL IZQ",
    }

    if tipo in etiquetas:
        cv2.putText(
            frame, etiquetas[tipo],
            (x, y + h + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1
        )

    if texto:
        cv2.putText(
            frame, texto,
            (x, max(14, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    return frame


def guardar_rostro_recortado(frame, nombre="persona", carpeta_base="dataset", tipo_esperado=None):
    """Guarda rostro recortado en dataset/nombre/"""
    from datetime import datetime

    vector, bbox, tipo = extraer_caracteristicas(frame, tipo_esperado=tipo_esperado)

    if bbox is None:
        return None, None, None, None

    x, y, w, h   = bbox
    h_img, w_img = frame.shape[:2]

    x1, y1 = max(0, x),         max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

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