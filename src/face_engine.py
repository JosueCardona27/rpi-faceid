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

# ─── MediaPipe Face Mesh — deteccion + angulos 3D precisos ───────────────────
_mp_face  = None
_mp_mesh  = None
_clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

def _get_mp():
    """Inicializa MediaPipe Face Mesh una sola vez."""
    global _mp_face, _mp_mesh
    if _mp_mesh is None:
        import mediapipe as mp
        _mp_face = mp.solutions.face_mesh
        _mp_mesh = _mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        print("[MP] MediaPipe Face Mesh iniciado.")
    return _mp_mesh


def _detectar_y_clasificar(frame):
    """
    Usa MediaPipe Face Mesh para:
      1. Detectar la cara y obtener bounding box
      2. Calcular angulos de cabeza con landmarks 3D reales

    Angulos calculados (estimacion PnP simplificada):
      - Yaw   (giro horizontal): positivo = cara girando a su izquierda
      - Pitch (inclinacion):     positivo = cara mirando hacia arriba

    Retorna: (bbox, tipo) donde bbox = (x,y,w,h) o None si no hay cara.
    """
    mesh = _get_mp()
    h_img, w_img = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = mesh.process(rgb)

    if not resultado.multi_face_landmarks:
        return None, None

    lm = resultado.multi_face_landmarks[0].landmark

    # ── Bounding box a partir de landmarks ───────────────────────────────────
    xs = [l.x * w_img for l in lm]
    ys = [l.y * h_img for l in lm]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    margen = int((x2 - x1) * 0.15)
    x1 = max(0, x1 - margen)
    y1 = max(0, y1 - margen)
    x2 = min(w_img, x2 + margen)
    y2 = min(h_img, y2 + margen)
    bbox = (x1, y1, x2 - x1, y2 - y1)

    # ── Calcular Yaw y Pitch con puntos clave ─────────────────────────────────
    # Landmarks clave (indices MediaPipe 468-point mesh):
    #   1  = punta nariz
    #   33 = comisura ojo izquierdo (lado derecho en imagen espejada)
    #   263 = comisura ojo derecho
    #   152 = menton
    #   10  = frente superior
    #   234 = mejilla izquierda
    #   454 = mejilla derecha

    def pt(idx):
        return np.array([lm[idx].x * w_img,
                         lm[idx].y * h_img,
                         lm[idx].z * w_img])

    nariz    = pt(1)
    ojo_izq  = pt(33)
    ojo_der  = pt(263)
    menton   = pt(152)
    frente   = pt(10)
    mejilla_izq = pt(234)
    mejilla_der = pt(454)

    # Centro de ojos
    centro_ojos = (ojo_izq + ojo_der) / 2.0

    # ── YAW: diferencia horizontal entre mejillas respecto a nariz ────────────
    # Si cara gira a su derecha → mejilla_der queda mas cerca / mejilla_izq mas lejos
    dist_izq = np.linalg.norm(nariz[:2] - mejilla_izq[:2])
    dist_der = np.linalg.norm(nariz[:2] - mejilla_der[:2])
    total_d  = dist_izq + dist_der + 1e-6
    # yaw_ratio > 0 → mejilla izq mas lejos → cara girada a su derecha
    # yaw_ratio < 0 → mejilla der mas lejos → cara girada a su izquierda
    yaw_ratio = (dist_izq - dist_der) / total_d

    # ── PITCH: posicion vertical de la nariz relativa a ojos y menton ─────────
    # Normalizar por altura cara
    altura_cara = abs(menton[1] - frente[1]) + 1e-6
    # Posicion normalizada de nariz entre ojos(0) y menton(1)
    pos_nariz_v = (nariz[1] - centro_ojos[1]) / altura_cara
    # pos_nariz_v frontal ≈ 0.4
    # cara hacia ABAJO → menton desaparece, nariz sube → pos < 0.35
    # cara hacia ARRIBA → frente desaparece, nariz baja → pos > 0.50

    # ── Clasificacion ─────────────────────────────────────────────────────────
    UMBRAL_YAW   = 0.12   # >12% asimetria → perfil
    UMBRAL_ABAJO = 0.34   # nariz muy alta en cara → mirando hacia abajo
    UMBRAL_ARRIBA= 0.50   # nariz muy baja en cara → mirando hacia arriba

    if yaw_ratio > UMBRAL_YAW:
        return bbox, TIPO_PERFIL_D
    elif yaw_ratio < -UMBRAL_YAW:
        return bbox, TIPO_PERFIL_I
    elif pos_nariz_v < UMBRAL_ABAJO:
        return bbox, TIPO_ABAJO
    elif pos_nariz_v > UMBRAL_ARRIBA:
        return bbox, TIPO_ABAJO   # arriba extremo tb cuenta como ABAJO
    else:
        return bbox, TIPO_FRONTAL


# ─── DNN fallback (si MediaPipe no esta disponible) ──────────────────────────
_dnn_net = None

def _encontrar_dnn():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
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
        else:
            _dnn_net = False
    return _dnn_net if _dnn_net else None

def _detectar_dnn_fallback(frame, conf_min=0.55):
    net = _get_dnn()
    if net is None:
        return None
    h_img, w_img = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    dets = net.forward()
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf >= conf_min:
            x1 = max(0, int(dets[0,0,i,3]*w_img))
            y1 = max(0, int(dets[0,0,i,4]*h_img))
            x2 = min(w_img, int(dets[0,0,i,5]*w_img))
            y2 = min(h_img, int(dets[0,0,i,6]*h_img))
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2-x1, y2-y1)
    return None


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


# ─── definicion de zonas sobre imagen 128x128 ─────────────────────────────────
#  (r0, r1, c0, c1, nombre)
ZONAS = [
    (0,  40,  0,  128, "frente"),
    (28, 65,  0,   58, "ojo_izq"),
    (28, 65, 70,  128, "ojo_der"),
    (55, 92, 28,  100, "nariz"),
    (62, 100, 0,   50, "mejilla_izq"),
    (62, 100, 78, 128, "mejilla_der"),
    (88, 128, 14, 114, "boca_menton"),
]

N_ZONAS  = len(ZONAS)   # 7
LBP_BINS = 59
VECTOR_DIM = N_ZONAS * LBP_BINS  # 7 × 59 = 413

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
    Usa MediaPipe Face Mesh para detectar la cara y calcular angulos 3D reales.
    Si MediaPipe falla usa DNN como fallback (sin clasificacion de angulo).

    Retorna: (vector, pesos, coords, tipo)  o  (None, None, None, None)
    """
    # ── Intentar MediaPipe primero ────────────────────────────────────────────
    try:
        bbox, tipo = _detectar_y_clasificar(frame)
    except Exception as e:
        print(f"[MP] Error: {e} — usando DNN fallback")
        bbox = _detectar_dnn_fallback(frame)
        tipo = TIPO_FRONTAL if bbox else None

    if bbox is None:
        return None, None, None, None

    x, y, w, h = bbox
    h_img, w_img = frame.shape[:2]

    # ── Recortar cara y extraer LBP ───────────────────────────────────────────
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x+w), min(h_img, y+h)
    gris    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recorte = preprocesar_cara(gris[y1:y2, x1:x2])
    if recorte.size == 0:
        return None, None, None, None

    cara128 = cv2.resize(recorte, (128, 128))

    hists, pesos = [], []
    for r0, r1, c0, c1, _ in ZONAS:
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