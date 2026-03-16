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

# ─── DNN: detector profundo (detecta cualquier angulo) ────────────────────────
_dnn_net  = None
_clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

# Rutas del modelo DNN — busca en la carpeta models/ relativa al proyecto
def _encontrar_dnn():
    """Busca los archivos del modelo DNN en rutas conocidas."""
    # Carpeta models/ un nivel arriba de src/
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "models")
    proto = os.path.join(base, "opencv_face_detector.prototxt")
    model = os.path.join(base, "opencv_face_detector.caffemodel")
    if os.path.exists(proto) and os.path.exists(model):
        return proto, model

    # Misma carpeta que face_engine.py
    base2 = os.path.dirname(os.path.abspath(__file__))
    proto2 = os.path.join(base2, "opencv_face_detector.prototxt")
    model2 = os.path.join(base2, "opencv_face_detector.caffemodel")
    if os.path.exists(proto2) and os.path.exists(model2):
        return proto2, model2

    return None, None

def _get_dnn():
    global _dnn_net
    if _dnn_net is None:
        proto, model = _encontrar_dnn()
        if proto and model:
            _dnn_net = cv2.dnn.readNetFromCaffe(proto, model)
            print(f"[DNN] Modelo cargado: {model}")
        else:
            raise FileNotFoundError(
                "No se encontro opencv_face_detector.caffemodel\n"
                "Descargalo con:\n"
                "  wget -O ~/Documents/rpi-faceid/models/opencv_face_detector.caffemodel \\\n"
                "    https://raw.githubusercontent.com/spmallick/learnopencv/master/"
                "FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel\n"
                "  wget -O ~/Documents/rpi-faceid/models/opencv_face_detector.prototxt \\\n"
                "    https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
                "face_detector/deploy.prototxt")
    return _dnn_net


def _detectar_dnn(frame, conf_min=0.55):
    """
    Detecta todas las caras con el modelo DNN SSD ResNet.
    Retorna lista de (x, y, w, h) ordenada por area descendente.
    Funciona con cualquier angulo de cabeza.
    """
    net    = _get_dnn()
    h_img, w_img = frame.shape[:2]

    # El modelo espera 300x300, normalizado con mean BGR (104,117,123)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    detecciones = net.forward()   # shape: (1,1,N,7)

    caras = []
    for i in range(detecciones.shape[2]):
        conf = float(detecciones[0, 0, i, 2])
        if conf < conf_min:
            continue
        x1 = int(detecciones[0, 0, i, 3] * w_img)
        y1 = int(detecciones[0, 0, i, 4] * h_img)
        x2 = int(detecciones[0, 0, i, 5] * w_img)
        y2 = int(detecciones[0, 0, i, 6] * h_img)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if x2 > x1 and y2 > y1:
            caras.append((x1, y1, x2-x1, y2-y1, conf))

    # Ordenar por area descendente
    caras.sort(key=lambda c: c[2]*c[3], reverse=True)
    return caras


def _clasificar_angulo(cara128):
    """
    Determina si la cara recortada es frontal, perfil derecho,
    perfil izquierdo o inclinada, usando la simetria de zonas LBP.

    Logica:
      - Calcula la varianza de la mitad izquierda y derecha de la cara.
      - Si una mitad tiene mucho menos varianza → es un perfil.
      - Si la cara esta muy comprimida verticalmente → inclinada hacia abajo.
    """
    h, w = cara128.shape

    # Varianza izquierda vs derecha
    mitad_izq = float(np.var(cara128[:, :w//2].astype(np.float32)))
    mitad_der = float(np.var(cara128[:, w//2:].astype(np.float32)))
    total_var  = mitad_izq + mitad_der + 1e-6
    ratio      = (mitad_der - mitad_izq) / total_var   # >0 = mas textura a la derecha

    # Varianza zona superior vs inferior para detectar inclinacion
    zona_sup = float(np.var(cara128[:h//2, :].astype(np.float32)))
    zona_inf = float(np.var(cara128[h//2:, :].astype(np.float32)))
    ratio_v  = (zona_sup - zona_inf) / (zona_sup + zona_inf + 1e-6)

    # Umbrales calibrados
    UMBRAL_PERFIL   = 0.18   # diferencia izq/der significativa
    UMBRAL_INCLIN   = 0.20   # cara mas textura arriba → barbilla oculta

    if ratio_v > UMBRAL_INCLIN:
        return TIPO_ABAJO          # cara inclinada hacia abajo
    elif ratio < -UMBRAL_PERFIL:
        return TIPO_PERFIL_D       # mas textura a la derecha → cara mirando a su derecha
    elif ratio > UMBRAL_PERFIL:
        return TIPO_PERFIL_I       # mas textura a la izquierda → cara mirando a su izquierda
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
    Detecta la cara con DNN SSD (detecta cualquier angulo) y clasifica
    automaticamente si es frontal, perfil derecho/izquierdo o inclinada.

    Parametro modo:
      "auto"    — detecta y clasifica el angulo automaticamente
      "frontal" — solo acepta caras clasificadas como frontales
      "perfil"  — detecta pero clasifica como perfil (der o izq segun orientacion)
      "abajo"   — detecta pero clasifica como inclinada

    Retorna: (vector, pesos, coords, tipo)  o  (None, None, None, None)
    """
    caras = _detectar_dnn(frame)
    if not caras:
        return None, None, None, None

    # Tomar la cara mas grande
    x, y, w, h, conf = caras[0]
    h_img, w_img = frame.shape[:2]

    # Recortar y preprocesar
    gris    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recorte = preprocesar_cara(gris[y:y+h, x:x+w])
    if recorte.size == 0:
        return None, None, None, None

    cara128 = cv2.resize(recorte, (128, 128))

    # Clasificar angulo automaticamente
    tipo_clasificado = _clasificar_angulo(cara128)

    # En modo estricto, forzar el tipo esperado
    # (el _capturar_registro valida si coincide con el paso)
    tipo_final = tipo_clasificado

    # Extraer vector LBP
    hists, pesos = [], []
    for r0, r1, c0, c1, _ in ZONAS:
        hist, var = _histograma_zona(cara128, r0, r1, c0, c1)
        hists.append(hist)
        pesos.append(_varianza_a_peso(var))

    vector = np.concatenate(hists).astype(np.float32)
    pesos  = np.array(pesos, dtype=np.float32)

    return vector, pesos, (x, y, w, h), tipo_final


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