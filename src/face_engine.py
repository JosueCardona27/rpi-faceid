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


# ─── Clasificacion de angulo con solvePnP ────────────────────────────────────
# Puntos 3D canonicos de una cara promedio (en mm, origen en nariz)
# Basados en el modelo de cara estandar de OpenCV head-pose estimation
_PUNTOS_3D = np.array([
    [ 0.0,    0.0,    0.0  ],   # Punta de la nariz
    [ 0.0,   -63.6,  -12.5],   # Menton
    [-43.3,   32.7,  -26.0],   # Comisura ojo izquierdo
    [ 43.3,   32.7,  -26.0],   # Comisura ojo derecho
    [-28.9,  -28.9,  -24.1],   # Comisura boca izquierda
    [ 28.9,  -28.9,  -24.1],   # Comisura boca derecha
], dtype=np.float64)

# Detector de landmarks faciales de OpenCV (5 puntos: ojos, nariz, boca)
_lm_detector = None

def _get_lm_detector():
    """Carga el detector de landmarks de OpenCV."""
    global _lm_detector
    if _lm_detector is None:
        # Buscar el modelo de landmarks en rutas conocidas
        nombres = [
            "face_landmark_model.dat",
            "lbfmodel.yaml",
        ]
        rutas_base = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
            os.path.dirname(os.path.abspath(__file__)),
            "/usr/share/opencv4",
            "/usr/local/share/opencv4",
        ]
        for base in rutas_base:
            for nombre in nombres:
                ruta = os.path.join(base, nombre)
                if os.path.exists(ruta):
                    try:
                        det = cv2.face.createFacemarkLBF()
                        det.loadModel(ruta)
                        _lm_detector = det
                        print(f"[LM] Landmarks cargados: {ruta}")
                        return _lm_detector
                    except Exception:
                        pass
        _lm_detector = False
        print("[LM] Landmarks no disponibles — usando clasificacion por asimetria")
    return _lm_detector if _lm_detector else None


def _clasificar_con_solvepnp(cara_recortada_gris, bbox, frame_shape):
    """
    Intenta calcular yaw/pitch con solvePnP si hay landmarks disponibles.
    Si no, usa clasificacion por asimetria de bordes (metodo robusto).
    Retorna tipo: TIPO_FRONTAL / TIPO_PERFIL_D / TIPO_PERFIL_I / TIPO_ABAJO
    """
    lm_det = _get_lm_detector()
    x, y, w, h = bbox
    fw, fh = frame_shape[1], frame_shape[0]

    if lm_det is not None:
        # ── Metodo solvePnP con landmarks ─────────────────────────────────────
        try:
            rect = np.array([[x, y, w, h]], dtype=np.int32)
            ok, landmarks = lm_det.fit(cara_recortada_gris, rect)
            if ok and len(landmarks) > 0:
                lm = landmarks[0][0]   # shape (5,2) o (68,2)
                # Usar los 5 puntos basicos si estan disponibles
                if len(lm) >= 6:
                    pts_2d = lm[:6].astype(np.float64)
                elif len(lm) >= 5:
                    # 5 puntos: ojo_izq, ojo_der, nariz, boca_izq, boca_der
                    nariz   = lm[2]
                    ojo_izq = lm[0]
                    ojo_der = lm[1]
                    boca_izq = lm[3]
                    boca_der = lm[4]
                    # Estimar menton y nariz punta para completar 6 puntos
                    menton = nariz + (nariz - (ojo_izq + ojo_der) / 2) * 1.5
                    pts_2d = np.array([nariz, menton, ojo_izq, ojo_der,
                                       boca_izq, boca_der], dtype=np.float64)
                else:
                    raise ValueError("Pocos landmarks")

                focal = fw
                centro = (fw / 2, fh / 2)
                cam_matrix = np.array([
                    [focal, 0,     centro[0]],
                    [0,     focal, centro[1]],
                    [0,     0,     1        ]
                ], dtype=np.float64)
                dist_coefs = np.zeros((4, 1))

                ok2, rvec, tvec = cv2.solvePnP(
                    _PUNTOS_3D, pts_2d, cam_matrix, dist_coefs,
                    flags=cv2.SOLVEPNP_ITERATIVE)

                if ok2:
                    rmat, _ = cv2.Rodrigues(rvec)
                    # Extraer yaw y pitch en grados
                    pitch = float(np.degrees(np.arcsin(-rmat[2, 0])))
                    yaw   = float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2])))

                    # Umbrales en grados
                    if yaw > 18:
                        return TIPO_PERFIL_D
                    elif yaw < -18:
                        return TIPO_PERFIL_I
                    elif pitch < -15:
                        return TIPO_ABAJO
                    elif pitch > 20:
                        return TIPO_ABAJO
                    else:
                        return TIPO_FRONTAL
        except Exception:
            pass

    # ── Fallback: asimetria de bordes sobre la cara recortada ─────────────────
    return _clasificar_por_asimetria(cara_recortada_gris)


def _clasificar_por_asimetria(cara128):
    """
    Clasificacion robusta usando asimetria de bordes Sobel.
    Funciona sin landmarks — solo con la imagen de la cara 128x128.

    Logica:
      YAW  : Si volteas a la derecha, la mejilla izquierda ocupa mas espacio
             → mas bordes en la mitad izquierda de la imagen
      PITCH: Si bajas la cabeza, los ojos/frente ocupan mas espacio
             → mas bordes en la zona superior
    """
    # Calcular mapa de bordes
    gx = cv2.Sobel(cara128, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(cara128, cv2.CV_32F, 0, 1, ksize=3)
    bordes = np.abs(gx) + np.abs(gy)

    h, w = bordes.shape

    # Dividir en mitades izq/der con zona central ignorada (10% central)
    margen = w // 10
    izq = float(np.mean(bordes[:, :w//2 - margen]))
    der = float(np.mean(bordes[:, w//2 + margen:]))
    total_h = izq + der + 1e-6
    # ratio > 0: mas bordes izquierda → cara girada a su DERECHA
    # ratio < 0: mas bordes derecha  → cara girada a su IZQUIERDA
    ratio_h = (izq - der) / total_h

    # Dividir en 3 bandas verticales (ignorar zona media)
    t1 = float(np.mean(bordes[:h//3,    :]))   # frente/ojos
    t2 = float(np.mean(bordes[h//3:2*h//3, :]))  # nariz/mejillas
    t3 = float(np.mean(bordes[2*h//3:,  :]))   # boca/menton/cuello
    total_v = t1 + t2 + t3 + 1e-6
    # ratio_v > 0: mas info arriba → cara inclinada hacia ABAJO
    ratio_v = (t1 - t3) / total_v

    # Umbrales — mas permisivos que antes
    UMBRAL_YAW   = 0.10
    UMBRAL_ABAJO = 0.12

    if ratio_h > UMBRAL_YAW:
        return TIPO_PERFIL_D
    elif ratio_h < -UMBRAL_YAW:
        return TIPO_PERFIL_I
    elif ratio_v > UMBRAL_ABAJO:
        return TIPO_ABAJO
    elif ratio_v < -UMBRAL_ABAJO:
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
    gris    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recorte = preprocesar_cara(gris[y1:y2, x1:x2])
    if recorte.size == 0:
        return None, None, None, None

    cara128 = cv2.resize(recorte, (128, 128))

    # Clasificar angulo
    tipo = _clasificar_con_solvepnp(gris, (x1, y1, x2-x1, y2-y1), frame.shape)

    # Extraer vector LBP
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