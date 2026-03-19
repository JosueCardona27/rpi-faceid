"""
face_engine.py - Versión optimizada sin face_recognition
Sistema de detección frontal + lateral con Haar + Sobel puro
"""
import os
import cv2
import numpy as np

TIPO_FRONTAL = "frontal"
TIPO_PERFIL_D = "perfil_der"
TIPO_PERFIL_I = "perfil_izq"

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# =============================================================================
#  DETECTORES HAAR
# =============================================================================

_haar_frontal = None
_haar_perfil = None
_det_init = False


def _init_detectores():
    """Inicializa cascadas Haar."""
    global _haar_frontal, _haar_perfil, _det_init

    if _det_init:
        return
    _det_init = True

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

    # Frontal
    rutas_frontal = [
        os.path.join(base, "haarcascade_frontalface_default.xml"),
    ]
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        rutas_frontal.append(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for ruta in rutas_frontal:
        if os.path.exists(ruta):
            _haar_frontal = cv2.CascadeClassifier(ruta)
            if not _haar_frontal.empty():
                print(f"[DET] Haar frontal: {ruta}")
                break

    # Perfil
    rutas_perfil = [
        os.path.join(base, "haarcascade_profileface.xml"),
    ]
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        rutas_perfil.append(cv2.data.haarcascades + "haarcascade_profileface.xml")

    for ruta in rutas_perfil:
        if os.path.exists(ruta):
            _haar_perfil = cv2.CascadeClassifier(ruta)
            if not _haar_perfil.empty():
                print(f"[DET] Haar perfil: {ruta}")
                break

    if _haar_frontal is None:
        print("[DET] ⚠️  Haar frontal NO ENCONTRADO")
    if _haar_perfil is None:
        print("[DET] ⚠️  Haar perfil NO ENCONTRADO - usando frontal relajado")


def _detectar_caras(frame, tipo_esperado=None):
    """
    Detecta caras con estrategia adaptativa.
    - Si tipo_esperado es FRONTAL: prioriza frontal
    - Si tipo_esperado es PERFIL_*: prioriza perfiles
    """
    _init_detectores()
    
    h_img, w_img = frame.shape[:2]
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = _clahe.apply(gris)
    
    resultados = []
    
    # ESTRATEGIA: Si esperamos FRONTAL
    if tipo_esperado == TIPO_FRONTAL:
        if _haar_frontal is not None:
            caras = _haar_frontal.detectMultiScale(
                gris,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(50, 50)
            )
            for (x, y, w, h) in caras:
                resultados.append((int(x), int(y), int(w), int(h), 0.85))
        
        # Fallback: perfiles como último recurso
        if not resultados and _haar_perfil is not None:
            caras = _haar_perfil.detectMultiScale(
                gris,
                scaleFactor=1.02,
                minNeighbors=2,
                minSize=(30, 30)
            )
            for (x, y, w, h) in caras:
                resultados.append((int(x), int(y), int(w), int(h), 0.70))
    
    # ESTRATEGIA: Si esperamos PERFIL
    elif tipo_esperado in (TIPO_PERFIL_D, TIPO_PERFIL_I):
        # Intentar perfil directo
        if _haar_perfil is not None:
            gris_det = cv2.flip(gris, 1) if tipo_esperado == TIPO_PERFIL_I else gris
            
            caras = _haar_perfil.detectMultiScale(
                gris_det,
                scaleFactor=1.02,
                minNeighbors=2,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in caras:
                if tipo_esperado == TIPO_PERFIL_I:
                    x = w_img - x - w
                resultados.append((int(x), int(y), int(w), int(h), 0.90))
        
        # Fallback: frontal relajado
        if not resultados and _haar_frontal is not None:
            caras = _haar_frontal.detectMultiScale(
                gris,
                scaleFactor=1.03,
                minNeighbors=3,
                minSize=(40, 40)
            )
            for (x, y, w, h) in caras:
                resultados.append((int(x), int(y), int(w), int(h), 0.75))
    
    # SIN TIPO ESPERADO: Buscar ambos
    else:
        if _haar_frontal is not None:
            caras = _haar_frontal.detectMultiScale(
                gris,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(50, 50)
            )
            for (x, y, w, h) in caras:
                resultados.append((int(x), int(y), int(w), int(h), 0.85))
        
        if _haar_perfil is not None:
            caras = _haar_perfil.detectMultiScale(
                gris,
                scaleFactor=1.02,
                minNeighbors=2,
                minSize=(30, 30)
            )
            for (x, y, w, h) in caras:
                resultados.append((int(x), int(y), int(w), int(h), 0.80))
            
            # Perfil invertido
            gris_flip = cv2.flip(gris, 1)
            caras = _haar_perfil.detectMultiScale(
                gris_flip,
                scaleFactor=1.02,
                minNeighbors=2,
                minSize=(30, 30)
            )
            for (x, y, w, h) in caras:
                x = w_img - x - w
                resultados.append((int(x), int(y), int(w), int(h), 0.80))
    
    # Filtrar y ordenar por tamaño
    resultados = [c for c in resultados if c[2] > 15 and c[3] > 15]
    resultados = sorted(resultados, key=lambda c: c[2] * c[3], reverse=True)
    
    return resultados


# =============================================================================
#  CLASIFICACIÓN DE ÁNGULO - SOBEL MEJORADO
# =============================================================================

_buf_yaw = []
_BUF_N_YAW = 5


def _calcular_yaw_sobel(frame_gris, bbox):
    """
    Calcula YAW (rotación horizontal) usando Sobel en 3 regiones.
    - Izquierda, Centro, Derecha
    - Compara densidad de bordes
    """
    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_gris.shape[1], x + w)
    y2 = min(frame_gris.shape[0], y + h)
    
    recorte = frame_gris[y1:y2, x1:x2]
    if recorte.size == 0:
        return 0.0
    
    try:
        # Redimensionar a 128x128 para consistencia
        cara = cv2.resize(recorte, (128, 128))
        
        # Aplicar Sobel X (detecta cambios horizontales = perfil)
        gx = cv2.Sobel(cara, cv2.CV_32F, 1, 0, ksize=7)
        gx_abs = np.abs(gx)
        
        # Dividir en 3 columnas
        w_col = cara.shape[1] // 3
        
        izq = np.mean(gx_abs[:, :w_col])
        cen = np.mean(gx_abs[:, w_col:2*w_col])
        der = np.mean(gx_abs[:, 2*w_col:])
        
        # Si izquierda tiene más bordes: usuario mira a la izquierda (PERFIL_IZQ)
        # Si derecha tiene más bordes: usuario mira a la derecha (PERFIL_DER)
        asimetria = (der - izq) / (der + izq + 1e-6)
        
        # Escalar a ±120
        yaw = asimetria * 100.0
        
        return yaw
    except Exception as e:
        print(f"[Sobel] Error: {e}")
        return 0.0


def _clasificar_angulo(frame, bbox, frame_shape, tipo_esperado=None):
    """
    Clasifica si es FRONTAL o PERFIL_D o PERFIL_I.
    Si tipo_esperado está dado, lo respeta.
    """
    global _buf_yaw
    
    # Si ya se indicó el tipo, respetarlo
    if tipo_esperado in (TIPO_FRONTAL, TIPO_PERFIL_D, TIPO_PERFIL_I):
        return tipo_esperado
    
    try:
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yaw = _calcular_yaw_sobel(frame_gris, bbox)
    except Exception:
        yaw = 0.0
    
    # Buffer con mediana
    _buf_yaw.append(yaw)
    if len(_buf_yaw) > _BUF_N_YAW:
        _buf_yaw.pop(0)
    
    yaw_suavizado = float(np.median(_buf_yaw)) if _buf_yaw else yaw
    
    # Umbrales: ±25 es bastante generoso para perfiles
    if yaw_suavizado > 25.0:
        return TIPO_PERFIL_D
    elif yaw_suavizado < -25.0:
        return TIPO_PERFIL_I
    else:
        return TIPO_FRONTAL


def _extraer_angulos_lbf(gris, bbox, fw, fh):
    """Compatibilidad - retorna None (no disponible sin landmarks)"""
    return None, None


# =============================================================================
#  ZONAS LBP
# =============================================================================

ZONAS_FRONTAL = [
    (0, 40, 0, 128, "frente"),
    (28, 65, 0, 58, "ojo_izq"),
    (28, 65, 70, 128, "ojo_der"),
    (55, 92, 28, 100, "nariz"),
    (62, 100, 0, 50, "mejilla_izq"),
    (62, 100, 78, 128, "mejilla_der"),
    (88, 128, 14, 114, "boca_menton"),
    (20, 100, 0, 128, "cara_media"),
]

ZONAS_PERFIL_D = [
    (0, 40, 0, 128, "frente"),
    (20, 60, 0, 60, "ojo"),
    (45, 85, 0, 65, "nariz_lat"),
    (55, 95, 0, 55, "mejilla"),
    (65, 110, 0, 50, "mandibula"),
    (25, 75, 0, 40, "pomulo"),
    (82, 128, 0, 75, "menton"),
    (15, 105, 0, 80, "perfil_media"),
]

ZONAS_PERFIL_I = [
    (0, 40, 0, 128, "frente"),
    (20, 60, 68, 128, "ojo"),
    (45, 85, 63, 128, "nariz_lat"),
    (55, 95, 73, 128, "mejilla"),
    (65, 110, 78, 128, "mandibula"),
    (25, 75, 88, 128, "pomulo"),
    (82, 128, 53, 128, "menton"),
    (15, 105, 48, 128, "perfil_media"),
]

ZONAS_POR_TIPO = {
    TIPO_FRONTAL: ZONAS_FRONTAL,
    TIPO_PERFIL_D: ZONAS_PERFIL_D,
    TIPO_PERFIL_I: ZONAS_PERFIL_I,
}

ZONAS = ZONAS_FRONTAL
N_ZONAS = 8
LBP_BINS = 64
VECTOR_DIM = N_ZONAS * LBP_BINS


# =============================================================================
#  LBP
# =============================================================================

_UNIFORM_MAP = None


def _build_uniform_map():
    umap = np.full(256, 58, dtype=np.int32)
    idx = 0
    for code in range(256):
        b = format(code, "08b")
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
    img = gris.astype(np.int32)
    pad = np.pad(img, 1, mode="edge")
    h, w = gris.shape
    center = pad[1:-1, 1:-1]
    nbrs = [
        pad[0:-2, 0:-2], pad[0:-2, 1:-1], pad[0:-2, 2:],
        pad[1:-1, 2:], pad[2:, 2:], pad[2:, 1:-1],
        pad[2:, 0:-2], pad[1:-1, 0:-2],
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
    umap = _get_uniform_map()
    hist59 = np.bincount(
        umap[lbp_map.flatten()],
        minlength=59
    ).astype(np.float32)
    
    total = hist59.sum()
    if total > 0:
        hist59 /= total
    
    hist64 = np.zeros(LBP_BINS, dtype=np.float32)
    hist64[:59] = hist59
    return hist64


# =============================================================================
#  API PÚBLICA
# =============================================================================

def preprocesar_cara(gris_zona):
    return cv2.GaussianBlur(_clahe.apply(gris_zona), (3, 3), 0)


def extraer_caracteristicas(frame, haar_path=None, modo="auto", tipo_esperado=None):
    """
    Detecta cara, clasifica ángulo y extrae vector LBP.
    """
    caras = _detectar_caras(frame, tipo_esperado=tipo_esperado)
    if not caras:
        return None, None, None
    
    x, y, w, h, _ = caras[0]
    h_img, w_img = frame.shape[:2]
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    
    gris_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recorte = gris_full[y1:y2, x1:x2]
    
    if recorte.size == 0:
        return None, None, None
    
    recorte = preprocesar_cara(recorte)
    cara128 = cv2.resize(recorte, (128, 128))
    
    tipo = _clasificar_angulo(
        frame,
        (x1, y1, x2 - x1, y2 - y1),
        frame.shape,
        tipo_esperado=tipo_esperado
    )
    
    zonas = ZONAS_POR_TIPO.get(tipo, ZONAS_FRONTAL)
    
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
        TIPO_FRONTAL: (0, 212, 255),
        TIPO_PERFIL_D: (255, 165, 0),
        TIPO_PERFIL_I: (0, 165, 255),
    }
    c = colores_tipo.get(tipo, color)
    
    for p1, p2 in [
        ((x, y), (x + L, y)), ((x, y), (x, y + L)),
        ((x + w, y), (x + w - L, y)), ((x + w, y), (x + w, y + L)),
        ((x, y + h), (x + L, y + h)), ((x, y + h), (x, y + h - L)),
        ((x + w, y + h), (x + w - L, y + h)), ((x + w, y + h), (x + w, y + h - L)),
    ]:
        cv2.line(frame, p1, p2, c, 2)
    
    etiquetas = {
        TIPO_FRONTAL: "FRONTAL",
        TIPO_PERFIL_D: "PERFIL DER",
        TIPO_PERFIL_I: "PERFIL IZQ",
    }
    
    if tipo in etiquetas:
        cv2.putText(
            frame,
            etiquetas[tipo],
            (x, y + h + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            c,
            1
        )
    
    if texto:
        cv2.putText(
            frame,
            texto,
            (x, max(14, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    
    return frame


def guardar_rostro_recortado(frame, nombre="persona", carpeta_base="dataset", tipo_esperado=None):
    """Guarda rostro recortado en dataset/nombre/"""
    from datetime import datetime
    
    vector, bbox, tipo = extraer_caracteristicas(frame, tipo_esperado=tipo_esperado)
    
    if bbox is None:
        return None, None, None, None
    
    x, y, w, h = bbox
    h_img, w_img = frame.shape[:2]
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    
    rostro = frame[y1:y2, x1:x2]
    if rostro.size == 0:
        return None, None, None, None
    
    ruta_dir = os.path.join(carpeta_base, nombre)
    os.makedirs(ruta_dir, exist_ok=True)
    
    marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archivo = f"{nombre}_{tipo}_{marca}.png"
    ruta = os.path.join(ruta_dir, archivo)
    
    cv2.imwrite(ruta, rostro)
    return ruta, vector, bbox, tipo