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

# ─── inicializar detector y CLAHE ─────────────────────────────────────────────
_detector = None
_clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

def _get_detector(haar_path="haarcascade_frontalface_default.xml"):
    global _detector
    if _detector is None:
        _detector = cv2.CascadeClassifier(haar_path)
    return _detector


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


def extraer_caracteristicas(frame, haar_path="haarcascade_frontalface_default.xml"):
    """
    Detecta la cara mas grande y extrae el vector de caracteristicas.

    Retorna:
        vector    : np.ndarray float32 de dim VECTOR_DIM (413) — histogramas LBP
        pesos     : np.ndarray float32 de dim N_ZONAS   (7)   — peso por zona
        coords    : (x, y, w, h) de la cara en el frame original
        o bien (None, None, None) si no se detecto cara.
    """
    det  = _get_detector(haar_path)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # preprocesar imagen completa antes de detectar
    gris_proc = preprocesar_cara(gris)

    caras = det.detectMultiScale(gris_proc, scaleFactor=1.1,
                                  minNeighbors=5, minSize=(70, 70))
    if len(caras) == 0:
        return None, None, None

    x, y, w, h = sorted(caras, key=lambda c: c[2]*c[3], reverse=True)[0]
    recorte    = gris_proc[y:y+h, x:x+w]
    cara128    = cv2.resize(recorte, (128, 128))

    hists  = []
    pesos  = []
    for r0, r1, c0, c1, _ in ZONAS:
        hist, var = _histograma_zona(cara128, r0, r1, c0, c1)
        hists.append(hist)
        pesos.append(_varianza_a_peso(var))

    vector = np.concatenate(hists).astype(np.float32)
    pesos  = np.array(pesos, dtype=np.float32)

    return vector, pesos, (x, y, w, h)


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


def dibujar_overlay(frame, coords, color, texto=""):
    x, y, w, h = coords
    L = max(18, w // 4)
    for p1, p2 in [
        ((x,   y),   (x+L, y)),     ((x,   y),   (x,   y+L)),
        ((x+w, y),   (x+w-L, y)),   ((x+w, y),   (x+w, y+L)),
        ((x,   y+h), (x+L, y+h)),   ((x,   y+h), (x,   y+h-L)),
        ((x+w, y+h), (x+w-L, y+h)), ((x+w, y+h), (x+w, y+h-L)),
    ]:
        cv2.line(frame, p1, p2, color, 2)
    if texto:
        cv2.putText(frame, texto, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame