"""
face_engine.py
==============
Motor de extraccion de vectores faciales robusto ante oclusiones
(lentes, cubrebocas, gorras, etc.)

Tecnica: LBP (Local Binary Patterns) por zonas con deteccion de oclusion.

Como funciona:
  1. CLAHE  → normaliza iluminacion local (mejor que equalizeHist)
  2. GaussianBlur → reduce ruido de sensor
  3. Se divide la cara en 7 ZONAS solapadas
  4. En cada zona se calcula un histograma LBP uniforme (59 bins)
  5. Se calcula el "peso" de cada zona segun su varianza local:
       varianza baja  → zona cubierta (ej. cubrebocas) → peso bajo
       varianza alta  → zona visible                    → peso alto
  6. Vector final: 7 × 59 = 413 valores de histograma
                 + 7 pesos de zona
                 = 420 valores en total

Al comparar dos personas:
  - Solo se usan las zonas donde AMBOS tienen peso > umbral
  - Si hay ≥ 2 zonas comparables → distancia chi² ponderada
  - Si hay < 2 zonas → no hay suficiente informacion

Zonas (sobre imagen 128×128):
  ┌──────────────────────────────┐
  │  Z0: frente (filas 0-38)     │  ← siempre visible
  ├──────────┬───────────────────┤
  │ Z1: ojo  │  Z2: ojo derecho  │  ← visible salvo gorra muy baja
  │ izquierdo│  (filas 30-65)    │
  ├──────────┴───────────────────┤
  │  Z3: nariz/entrecejo         │  ← visible con cubrebocas
  │  (filas 58-90, centro)       │
  ├────────────┬─────────────────┤
  │ Z4: mejilla│  Z5: mejilla    │  ← visible sin cubrebocas
  │ izquierda  │  derecha        │
  ├────────────┴─────────────────┤
  │  Z6: boca/menton             │  ← cubierta con cubrebocas
  │  (filas 88-128)              │
  └──────────────────────────────┘
"""

import cv2
import numpy as np
import os

# ─── tipos de deteccion ───────────────────────────────────────────────────────
TIPO_FRONTAL  = "frontal"
TIPO_PERFIL_D = "perfil_der"
TIPO_PERFIL_I = "perfil_izq"
TIPO_ABAJO    = "abajo"

_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

# ─── DNN detector (SSD ResNet) ───────────────────────────────────────────────
_dnn_net = None

def _encontrar_dnn():
    base  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    proto = os.path.join(base, "opencv_face_detector.prototxt")
    model = os.path.join(base, "opencv_face_detector.caffemodel")
    if os.path.exists(proto) and os.path.exists(model):
        return proto, model
    return None, None

def _get_dnn():
    global _dnn_net
    if _dnn_net is None:
        proto, model = _encontrar_dnn()
        if proto and model:
            _dnn_net = cv2.dnn.readNetFromCaffe(proto, model)
            print(f"[DNN] Modelo cargado: {model}")
        else:
            _dnn_net = False
            print("[DNN] WARN: modelo no encontrado")
    return _dnn_net if _dnn_net else None

def _detectar_dnn(frame, conf_min=0.45):
    """Detecta caras con DNN SSD. Retorna lista de (x,y,w,h,conf)."""
    net = _get_dnn()
    if net is None:
        return []
    h_img, w_img = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    dets = net.forward()
    caras = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < conf_min:
            continue
        x1 = max(0, int(dets[0,0,i,3] * w_img))
        y1 = max(0, int(dets[0,0,i,4] * h_img))
        x2 = min(w_img, int(dets[0,0,i,5] * w_img))
        y2 = min(h_img, int(dets[0,0,i,6] * h_img))
        if x2 > x1 and y2 > y1:
            caras.append((x1, y1, x2-x1, y2-y1, conf))
    caras.sort(key=lambda c: c[2]*c[3], reverse=True)
    return caras


# ─── Landmarks faciales con cv2.face LBF (68 puntos) ─────────────────────────
_lm_detector = None

def _get_lm_detector():
    """Carga el modelo LBF de landmarks. Se inicializa una sola vez."""
    global _lm_detector
    if _lm_detector is not None:
        return _lm_detector if _lm_detector is not False else None

    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "models", "lbfmodel.yaml")
    if os.path.exists(ruta):
        try:
            det = cv2.face.createFacemarkLBF()
            det.loadModel(ruta)
            _lm_detector = det
            print(f"[LM] cv2.face LBF cargado: {ruta}")
            return _lm_detector
        except Exception as e:
            print(f"[LM] Error cargando LBF: {e}")

    _lm_detector = False
    print("[LM] lbfmodel.yaml no disponible — usando asimetria de bordes")
    return None


# ─── Puntos 3D modelo de cara estandar (para solvePnP) ───────────────────────
# Indices en landmarks de 68 puntos:
#   30 = punta nariz, 8 = menton, 36 = ojo izq ext,
#   45 = ojo der ext, 48 = boca izq, 54 = boca der
_PTS_3D = np.array([
    [ 0.0,    0.0,    0.0   ],   # 30 punta nariz
    [ 0.0,  -63.6,  -12.5  ],   # 8  menton
    [-43.3,  32.7,  -26.0  ],   # 36 ojo izq ext
    [ 43.3,  32.7,  -26.0  ],   # 45 ojo der ext
    [-28.9, -28.9,  -24.1  ],   # 48 boca izq
    [ 28.9, -28.9,  -24.1  ],   # 54 boca der
], dtype=np.float64)

_IDX_LM = [30, 8, 36, 45, 48, 54]   # indices en el array de 68 landmarks

# Buffer de suavizado para estabilizar angulos entre frames
_BUFFER_N    = 4
_buf_yaw     = []
_buf_pitch   = []


def _clasificar_angulo(frame_gris, bbox, frame_shape):
    """
    Calcula yaw y pitch con cv2.face LBF + solvePnP.
    Si LBF no esta disponible, cae a asimetria de bordes.
    Retorna: TIPO_FRONTAL / TIPO_PERFIL_D / TIPO_PERFIL_I / TIPO_ABAJO
    """
    global _buf_yaw, _buf_pitch

    lm_det = _get_lm_detector()
    x, y, w, h = bbox
    fw = frame_shape[1]
    fh = frame_shape[0]

    if lm_det is not None:
        try:
            # LBF necesita imagen completa + rect en formato (x,y,w,h)
            rect = np.array([[x, y, w, h]], dtype=np.int32)
            ok, landmarks = lm_det.fit(frame_gris, rect)

            if ok and len(landmarks) > 0:
                lm = landmarks[0][0]   # shape: (68, 2)

                # Extraer los 6 puntos clave
                pts_2d = np.array(
                    [lm[i] for i in _IDX_LM], dtype=np.float64)

                # Matriz de camara estimada
                focal      = fw
                cam_matrix = np.array([
                    [focal, 0,     fw / 2],
                    [0,     focal, fh / 2],
                    [0,     0,     1     ]
                ], dtype=np.float64)

                ok2, rvec, tvec = cv2.solvePnP(
                    _PTS_3D, pts_2d, cam_matrix,
                    np.zeros((4, 1)),
                    flags=cv2.SOLVEPNP_ITERATIVE)

                if ok2:
                    rmat, _ = cv2.Rodrigues(rvec)
                    yaw   = float(np.degrees(
                        np.arctan2(rmat[1, 0], rmat[0, 0])))
                    pitch = float(np.degrees(
                        np.arctan2(-rmat[2, 0],
                                   np.sqrt(rmat[2,1]**2 + rmat[2,2]**2))))

                    # Suavizar con buffer
                    _buf_yaw.append(yaw)
                    _buf_pitch.append(pitch)
                    if len(_buf_yaw) > _BUFFER_N:
                        _buf_yaw.pop(0)
                        _buf_pitch.pop(0)

                    yaw_s   = float(np.mean(_buf_yaw))
                    pitch_s = float(np.mean(_buf_pitch))

                    # Pitch tiene prioridad SOLO si yaw es pequeño
                    # Esto evita que voltear lateral se confunda con abajo
                    if yaw_s > 15:
                        return TIPO_PERFIL_D
                    elif yaw_s < -15:
                        return TIPO_PERFIL_I
                    elif pitch_s < -25 or pitch_s > 25:
                        return TIPO_ABAJO
                    else:
                        return TIPO_FRONTAL
        except Exception as e:
            pass   # fallback a asimetria

    # ── Fallback: asimetria de bordes ────────────────────────────────────────
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_shape[1], x+w), min(frame_shape[0], y+h)
    recorte = frame_gris[y1:y2, x1:x2]
    if recorte.size == 0:
        return TIPO_FRONTAL
    cara128 = cv2.resize(recorte, (128, 128))
    return _clasificar_por_asimetria(cara128)


# ─── Fallback: asimetria de bordes Sobel ─────────────────────────────────────
_buf_rh = []
_buf_rv = []

def _clasificar_por_asimetria(cara128):
    global _buf_rh, _buf_rv

    gx = cv2.Sobel(cara128, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(cara128, cv2.CV_32F, 0, 1, ksize=3)
    bordes = np.abs(gx) + np.abs(gy)
    h, w   = bordes.shape

    margen  = int(w * 0.15)
    izq     = float(np.mean(bordes[:, :w//2 - margen]))
    der     = float(np.mean(bordes[:, w//2 + margen:]))
    ratio_h = (izq - der) / (izq + der + 1e-6)

    t1      = float(np.mean(bordes[:h//3,   :]))
    t3      = float(np.mean(bordes[2*h//3:, :]))
    ratio_v = (t1 - t3) / (t1 + t3 + 1e-6)

    _buf_rh.append(ratio_h)
    _buf_rv.append(ratio_v)
    if len(_buf_rh) > 5:
        _buf_rh.pop(0)
        _buf_rv.pop(0)

    rh = float(np.mean(_buf_rh))
    rv = float(np.mean(_buf_rv))

    # YAW tiene prioridad sobre PITCH
    # Si hay asimetria lateral significativa → es perfil, no abajo
    if rh > 0.14:
        return TIPO_PERFIL_D
    elif rh < -0.11:
        return TIPO_PERFIL_I
    elif abs(rv) > 0.18:   # umbral mas alto para evitar falsos abajo
        return TIPO_ABAJO
    else:
        return TIPO_FRONTAL


# ─── mapa LBP uniforme (se construye una sola vez) ───────────────────────────
_UNIFORM_MAP = None

def _build_uniform_map():
    """
    Mapea los 256 codigos LBP a 59 bins:
      - 58 patrones uniformes (con <= 2 transiciones de bit): bins 0-57
      - 1 bin para todos los patrones no uniformes: bin 58
    """
    umap = np.full(256, 58, dtype=np.int32)
    idx  = 0
    for code in range(256):
        b = format(code, '08b')
        transitions = sum(b[i] != b[(i + 1) % 8] for i in range(8))
        if transitions <= 2:
            umap[code] = idx
            idx += 1
    return umap

def _get_uniform_map():
    global _UNIFORM_MAP
    if _UNIFORM_MAP is None:
        _UNIFORM_MAP = _build_uniform_map()
    return _UNIFORM_MAP


# ─── calculo LBP rapido con numpy ─────────────────────────────────────────────
def _lbp_imagen(gris):
    """
    Calcula el mapa LBP de una imagen en escala de grises.
    Usa numpy para velocidad (sin loops Python por pixel).
    Retorna array uint8 del mismo tamaño.
    """
    img   = gris.astype(np.int32)
    pad   = np.pad(img, 1, mode='edge')
    h, w  = gris.shape

    center = pad[1:-1, 1:-1]

    # 8 vecinos en orden circular (empezando arriba-izquierda)
    nbrs = [
        pad[0:-2, 0:-2],   # NO
        pad[0:-2, 1:-1],   # N
        pad[0:-2, 2:],     # NE
        pad[1:-1, 2:],     # E
        pad[2:,   2:],     # SE
        pad[2:,   1:-1],   # S
        pad[2:,   0:-2],   # SO
        pad[1:-1, 0:-2],   # O
    ]

    lbp = np.zeros((h, w), dtype=np.uint8)
    for bit, nbr in enumerate(nbrs):
        lbp |= ((nbr >= center).astype(np.uint8) << bit)

    return lbp


def _histograma_zona(cara128, r0, r1, c0, c1):
    """
    Calcula histograma LBP uniforme (59 bins, normalizado) de una zona.
    Tambien retorna la varianza local como indicador de si hay oclusion.
    """
    zona    = cara128[r0:r1, c0:c1]
    if zona.size == 0:
        return np.zeros(59, dtype=np.float32), 0.0

    lbp_map = _lbp_imagen(zona)
    umap    = _get_uniform_map()

    hist = np.bincount(umap[lbp_map.flatten()], minlength=59).astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total

    varianza = float(np.var(zona.astype(np.float32)))
    return hist, varianza


# ─── definicion de zonas por angulo sobre imagen 128x128 ──────────────────────
#
# FRONTAL: cara simetrica, 7 zonas balanceadas
#   ┌──────────────────────────────┐
#   │  Z0: frente                  │
#   ├──────────┬───────────────────┤
#   │ Z1:ojo izq│ Z2: ojo der      │
#   ├──────────┴───────────────────┤
#   │  Z3: nariz/entrecejo         │
#   ├────────────┬─────────────────┤
#   │ Z4:mejilla │ Z5:mejilla der  │
#   ├────────────┴─────────────────┤
#   │  Z6: boca/menton             │
#   └──────────────────────────────┘
#
# PERFIL_DER (cara girando a su derecha = lado izquierdo en imagen):
#   Visible: frente, ojo izq, nariz izq, mejilla izq, mandibula izq
#   Zonas desplazadas al lado izquierdo de la imagen
#
# PERFIL_IZQ (cara girando a su izquierda = lado derecho en imagen):
#   Visible: frente, ojo der, nariz der, mejilla der, mandibula der
#   Zonas desplazadas al lado derecho de la imagen
#
# ABAJO (cara inclinada hacia abajo):
#   Visible: frente prominente, ojos, nariz superior
#   Zonas concentradas en parte superior

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
    # Cara girando a su derecha — lado izquierdo de imagen visible
    (0,  40,  0,  128, "frente"),          # frente siempre visible
    (20, 60,  0,   64, "ojo_visible"),     # ojo izquierdo (lado visible)
    (45, 85,  0,   70, "nariz_lateral"),   # nariz vista de lado
    (55, 100, 0,   60, "mejilla_visible"), # mejilla izquierda
    (70, 115, 0,   55, "mandibula"),       # mandibula/jawline visible
    (30, 80,  0,   45, "pomulo"),          # pomulo visible
    (85, 128, 0,   80, "cuello_menton"),   # menton/cuello
]

ZONAS_PERFIL_I = [
    # Cara girando a su izquierda — lado derecho de imagen visible
    (0,  40,  0,  128, "frente"),          # frente siempre visible
    (20, 60, 64,  128, "ojo_visible"),     # ojo derecho (lado visible)
    (45, 85, 58,  128, "nariz_lateral"),   # nariz vista de lado
    (55, 100, 68, 128, "mejilla_visible"), # mejilla derecha
    (70, 115, 73, 128, "mandibula"),       # mandibula/jawline visible
    (30, 80,  83, 128, "pomulo"),          # pomulo visible
    (85, 128, 48, 128, "cuello_menton"),   # menton/cuello
]

ZONAS_ABAJO = [
    # Cara inclinada hacia abajo — frente y ojos prominentes
    (0,  35,  0,  128, "frente_sup"),     # frente superior muy visible
    (10, 50,  0,   58, "frente_izq"),     # frente izquierda
    (10, 50, 70,  128, "frente_der"),     # frente derecha
    (30, 70,  0,   58, "ojo_izq"),        # ojo izquierdo
    (30, 70, 70,  128, "ojo_der"),        # ojo derecho
    (50, 90, 28,  100, "nariz_sup"),      # nariz superior visible
    (0,  40, 28,  100, "entrecejo"),      # entrecejo prominente
]

# Mapa de angulo → zonas
ZONAS_POR_TIPO = {
    TIPO_FRONTAL:  ZONAS_FRONTAL,
    TIPO_PERFIL_D: ZONAS_PERFIL_D,
    TIPO_PERFIL_I: ZONAS_PERFIL_I,
    TIPO_ABAJO:    ZONAS_ABAJO,
}

# ZONAS legacy para compatibilidad (igual que frontal)
ZONAS = ZONAS_FRONTAL

N_ZONAS    = len(ZONAS_FRONTAL)   # 7 — igual para todos los angulos
LBP_BINS   = 59
VECTOR_DIM = N_ZONAS * LBP_BINS   # 7 × 59 = 413

# umbral de varianza: zona con var < este valor probablemente ocluida
VAR_MIN  = 250.0
VAR_FULL = 800.0   # por encima de este valor → peso maximo 1.0

def _varianza_a_peso(var):
    """Convierte varianza local a peso [0, 1]."""
    if var < VAR_MIN:
        return 0.0
    if var >= VAR_FULL:
        return 1.0
    return (var - VAR_MIN) / (VAR_FULL - VAR_MIN)


# ─── API PUBLICA ──────────────────────────────────────────────────────────────

def preprocesar_cara(gris_zona):
    """CLAHE + blur sobre una region gris."""
    eq  = _clahe.apply(gris_zona)
    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    return blur


def extraer_caracteristicas(frame, haar_path=None, modo="auto"):
    """
    Detecta la cara con DNN SSD y clasifica el angulo con:
      1. solvePnP (si hay landmarks disponibles) — angulos 3D exactos
      2. Asimetria de bordes Sobel (fallback siempre disponible)

    Retorna: (vector, pesos, coords, tipo)  o  (None, None, None, None)
    """
    caras = _detectar_dnn(frame)
    if not caras:
        return None, None, None, None

    x, y, w, h, _ = caras[0]
    h_img, w_img = frame.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x+w), min(h_img, y+h)
    gris_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recorte   = preprocesar_cara(gris_full[y1:y2, x1:x2])
    if recorte.size == 0:
        return None, None, None, None

    cara128 = cv2.resize(recorte, (128, 128))

    # Clasificar angulo con LBF landmarks + solvePnP (o fallback asimetria)
    tipo = _clasificar_angulo(gris_full, (x1, y1, x2-x1, y2-y1), frame.shape)

    # Seleccionar zonas segun el angulo detectado
    zonas_activas = ZONAS_POR_TIPO.get(tipo, ZONAS_FRONTAL)

    # Extraer vector LBP con las zonas del angulo correspondiente
    hists, pesos = [], []
    for r0, r1, c0, c1, _ in zonas_activas:
        hist, var = _histograma_zona(cara128, r0, r1, c0, c1)
        hists.append(hist)
        pesos.append(_varianza_a_peso(var))

    vector = np.concatenate(hists).astype(np.float32)
    pesos  = np.array(pesos, dtype=np.float32)

    return vector, pesos, (x1, y1, x2-x1, y2-y1), tipo


def distancia_ponderada(v1, p1, v2, p2):
    """
    Distancia chi-cuadrado ponderada entre dos vectores zonales.

    Solo compara zonas donde AMBOS tienen peso > 0.
    Retorna (distancia, n_zonas_usadas).

    Distancia 0.0 = identicos, valores tipicos 0.0–2.0+
    """
    dist_total = 0.0
    peso_total = 0.0
    zonas_usadas = 0

    for z in range(N_ZONAS):
        w = min(p1[z], p2[z])   # solo usar la zona si AMBOS la tienen visible
        if w < 0.15:
            continue

        i0 = z * LBP_BINS
        i1 = i0 + LBP_BINS
        h1 = v1[i0:i1]
        h2 = v2[i0:i1]

        # chi-cuadrado: Σ (h1-h2)² / (h1+h2+ε)
        denom = h1 + h2 + 1e-7
        chi2  = float(np.sum((h1 - h2)**2 / denom))

        dist_total  += w * chi2
        peso_total  += w
        zonas_usadas += 1

    if zonas_usadas < 2 or peso_total < 0.1:
        return float('inf'), 0   # no hay suficiente informacion

    return dist_total / peso_total, zonas_usadas


def dibujar_overlay(frame, coords, color, texto="", tipo=None):
    x, y, w, h = coords
    L = max(18, w // 4)

    # color de esquinas segun tipo de deteccion
    color_esquinas = color
    if tipo == TIPO_PERFIL_D:
        color_esquinas = (255, 165, 0)    # naranja — perfil derecho
    elif tipo == TIPO_PERFIL_I:
        color_esquinas = (0, 165, 255)    # azul claro — perfil izquierdo
    elif tipo == TIPO_ABAJO:
        color_esquinas = (180, 0, 255)    # morado — inclinado abajo

    for p1, p2 in [
        ((x,   y),   (x+L, y)),     ((x,   y),   (x,   y+L)),
        ((x+w, y),   (x+w-L, y)),   ((x+w, y),   (x+w, y+L)),
        ((x,   y+h), (x+L, y+h)),   ((x,   y+h), (x,   y+h-L)),
        ((x+w, y+h), (x+w-L, y+h)), ((x+w, y+h), (x+w, y+h-L)),
    ]:
        cv2.line(frame, p1, p2, color_esquinas, 2)

    # etiqueta del tipo de deteccion
    tipo_txt = {
        TIPO_FRONTAL:  "FRONTAL",
        TIPO_PERFIL_D: "PERFIL DER",
        TIPO_PERFIL_I: "PERFIL IZQ",
        TIPO_ABAJO:    "INCLINADO",
    }.get(tipo, "")

    if tipo_txt:
        cv2.putText(frame, tipo_txt,
                    (x, y + h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_esquinas, 1)
    if texto:
        cv2.putText(frame, texto, (x, max(14, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame