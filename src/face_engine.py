"""
face_engine.py - Version Deep Learning (MobileFaceNet + YuNet)
==============================================================
CAMBIOS vs version LBP:

  RECONOCIMIENTO:
    LBP 512-dim histogramas  ->  MobileFaceNet ONNX 512-dim embeddings
    Distancia chi-cuadrado   ->  Distancia coseno (vectores L2-normalizados)
    Cara 128x128 en gris     ->  Cara 112x112 RGB alineada con 5 landmarks

  DETECCION / ANGULO (SIN CAMBIOS):
    YuNet ONNX + Haar fallback
    Clasificacion por asimetria de landmarks

MODELO REQUERIDO:
  Archivo: models/w600k_mbf.onnx
  Se descarga automaticamente al primer arranque (~16 MB).
  Si falla la descarga automatica:
    1. Descarga buffalo_sc.zip desde:
       https://github.com/deepinsight/insightface/releases/tag/v0.7
    2. Extrae w600k_mbf.onnx y colócalo en la carpeta models/

INSTALACION:
  pip install onnxruntime
"""

import os
import cv2
import numpy as np

# -- Tipos de angulo ----------------------------------------------------------
TIPO_FRONTAL  = "frontal"
TIPO_PERFIL_D = "perfil_der"
TIPO_PERFIL_I = "perfil_izq"

# -- CLAHE (para deteccion, no para reconocimiento) ---------------------------
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# -- Rutas base ----------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS   = os.path.join(_BASE_DIR, "..", "models")

# -- Dimension del vector de caracteristicas ----------------------------------
VECTOR_DIM = 512   # MobileFaceNet w600k_mbf.onnx -> 512 dims

# -- Filtros de calidad de deteccion (usados en diagnostico.py) ---------------
_SCORE_MINIMO_CARA = 0.45   # score YuNet minimo para considerar cara real
_LAPLACIAN_MIN     = 25.0   # varianza Laplaciano minima (fondo IR ~ 3-20, cara ~ 25-150)

# -- Template de alineacion facial (5 puntos -> 112x112) ----------------------
_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# =============================================================================
#  SUAVIZADO DE BOUNDING BOX  (elimina el temblor del recuadro)
# =============================================================================

class _BboxSmoother:
    """
    Suaviza las coordenadas del bounding box cuadro a cuadro mediante
    una Media Movil Exponencial (EMA) para eliminar el temblor visible.

    alpha : fraccion del frame NUEVO que se mezcla con el historico.
            0.0 -> recuadro completamente fijo (no sigue al rostro)
            1.0 -> sin suavizado (comportamiento original, tiembla)
            0.35 -> buen equilibrio: suave pero sigue movimientos normales

    Uso interno — se instancia una vez como _bbox_smoother y se aplica
    automaticamente dentro de extraer_caracteristicas().
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._prev = None          # (x, y, w, h) en flotante

    def update(self, bbox):
        """
        Recibe bbox crudo (x, y, w, h) o None si no hay cara.
        Devuelve bbox suavizado como tupla de enteros, o None.
        """
        if bbox is None:
            self._prev = None      # resetear al perder la cara
            return None

        new = tuple(float(v) for v in bbox)

        if self._prev is None:
            # primer frame con cara: aceptar sin mezcla
            self._prev = new
            return bbox

        a   = self.alpha
        sx  = a * new[0] + (1 - a) * self._prev[0]
        sy  = a * new[1] + (1 - a) * self._prev[1]
        sw  = a * new[2] + (1 - a) * self._prev[2]
        sh  = a * new[3] + (1 - a) * self._prev[3]
        self._prev = (sx, sy, sw, sh)

        return (int(round(sx)), int(round(sy)),
                int(round(sw)), int(round(sh)))

    def reset(self):
        self._prev = None


# Instancia global — comparte estado entre llamadas consecutivas
_bbox_smoother = _BboxSmoother(alpha=0.35)


# =============================================================================
#  DETECTOR YuNet + Haar  (identico a version anterior)
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

    for nombre in ("face_detection_yunet_2023mar.onnx", "face_detection_yunet.onnx"):
        yunet_path = os.path.join(_MODELS, nombre)
        if os.path.exists(yunet_path):
            try:
                _yunet = cv2.FaceDetectorYN.create(
                    yunet_path, "", (640, 480),
                    score_threshold=0.30,
                    nms_threshold=0.3,
                    top_k=5000
                )
                print(f"[DET] YuNet cargado: {yunet_path}")
            except Exception as e:
                print(f"[DET] YuNet error: {e}")
                _yunet = None
            break

    if _yunet is None:
        print(f"[DET] YuNet NO encontrado en {_MODELS}")

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
        print("[DET] Ningun detector disponible.")


_ultimo_face_yunet = None


# =============================================================================
#  DETECCION
# =============================================================================

def _detectar_caras_yunet(frame):
    global _ultimo_face_yunet

    h_img, w_img      = frame.shape[:2]
    _yunet.setInputSize((w_img, h_img))
    _, faces           = _yunet.detect(frame)
    _ultimo_face_yunet = None

    if faces is None or len(faces) == 0:
        return []

    detecciones = []
    for face in faces:
        x = int(face[0]); y = int(face[1])
        w = int(face[2]); h = int(face[3])
        score = float(face[14])
        x = max(0, x);  y = max(0, y)
        w = min(w, w_img - x);  h = min(h, h_img - y)
        if w < 15 or h < 15:
            continue
        detecciones.append((x, y, w, h, round(score, 3), face))

    if not detecciones:
        return []

    detecciones.sort(key=lambda d: d[2] * d[3], reverse=True)
    _ultimo_face_yunet = detecciones[0][5]

    return [(d[0], d[1], d[2], d[3], d[4]) for d in detecciones]


def _detectar_caras_haar(frame, tipo_esperado=None):
    global _ultimo_face_yunet
    _ultimo_face_yunet = None

    h_img, w_img = frame.shape[:2]
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = _clahe.apply(gris)
    resultados = []

    if tipo_esperado in (TIPO_FRONTAL, None):
        if _haar_frontal is not None:
            caras = _haar_frontal.detectMultiScale(
                gris, scaleFactor=1.03, minNeighbors=3, minSize=(40, 40)
            )
            for (x, y, w, h) in caras:
                resultados.append((int(x), int(y), int(w), int(h), 0.75))

    if tipo_esperado in (TIPO_PERFIL_D, None):
        if _haar_perfil is not None:
            caras = _haar_perfil.detectMultiScale(
                gris, scaleFactor=1.02, minNeighbors=2, minSize=(30, 30)
            )
            for (x, y, w, h) in caras:
                resultados.append((int(x), int(y), int(w), int(h), 0.70))

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
    _init_detectores()
    if _yunet is not None:
        return _detectar_caras_yunet(frame)
    return _detectar_caras_haar(frame, tipo_esperado)


# =============================================================================
#  CLASIFICACION DE ANGULO  (identica a version anterior)
# =============================================================================

_buf_yaw   = []
_BUF_N_YAW = 8


def _clasificar_angulo_con_landmarks(face_row):
    x   = float(face_row[0])
    w   = float(face_row[2])
    cx  = x + w / 2.0
    x_od = float(face_row[4])
    x_oi = float(face_row[6])
    x_n  = float(face_row[8])

    dist_od   = cx - x_od
    dist_oi   = x_oi - cx
    total     = abs(dist_od) + abs(dist_oi) + 1e-6
    asimetria = (dist_od - dist_oi) / total

    _buf_yaw.append(asimetria)
    if len(_buf_yaw) > _BUF_N_YAW:
        _buf_yaw.pop(0)
    asm = float(np.median(_buf_yaw))

    if asm > 0.40:
        return TIPO_PERFIL_D
    elif asm < -0.40:
        return TIPO_PERFIL_I

    desv = (x_n - cx) / (w + 1e-6)
    if desv > 0.15:
        return TIPO_PERFIL_I
    elif desv < -0.15:
        return TIPO_PERFIL_D

    return TIPO_FRONTAL


def _calcular_yaw_sobel(frame_gris, bbox):
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_gris.shape[1], x + w), min(frame_gris.shape[0], y + h)
    recorte = frame_gris[y1:y2, x1:x2]
    if recorte.size == 0:
        return 0.0
    try:
        cara = cv2.resize(recorte, (128, 128))
        gx   = cv2.Sobel(cara, cv2.CV_32F, 1, 0, ksize=7)
        gabs = np.abs(gx)
        wc   = cara.shape[1] // 3
        izq  = np.mean(gabs[:, :wc])
        der  = np.mean(gabs[:, 2*wc:])
        return (der - izq) / (der + izq + 1e-6) * 100.0
    except Exception:
        return 0.0


def _clasificar_angulo(frame, bbox, frame_shape, tipo_esperado=None):
    global _buf_yaw
    if _ultimo_face_yunet is not None:
        return _clasificar_angulo_con_landmarks(_ultimo_face_yunet)
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
    return TIPO_FRONTAL


def _extraer_angulos_lbf(gris, bbox, fw, fh):
    """Compatibilidad con diagnostico.py."""
    return None, None


# =============================================================================
#  RECONOCIMIENTO DEEP LEARNING  (MobileFaceNet ONNX)
# =============================================================================

_ort_session = None
_ort_input   = None
_recog_listo = False


def _descargar_modelo(dest_path):
    import urllib.request
    import zipfile

    url      = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
    zip_path = dest_path.replace("w600k_mbf.onnx", "buffalo_sc_tmp.zip")

    print("[RECOG] Descargando modelo MobileFaceNet (~16 MB)...")
    try:
        def _prog(count, block, total):
            mb = count * block / 1_048_576
            print(f"\r[RECOG] {mb:.1f} MB...", end="", flush=True)
        urllib.request.urlretrieve(url, zip_path, reporthook=_prog)
        print()

        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                if name.endswith("w600k_mbf.onnx"):
                    data = z.read(name)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    with open(dest_path, "wb") as f:
                        f.write(data)
                    print(f"[RECOG] Modelo guardado: {dest_path}")
                    break
            else:
                print("[RECOG] w600k_mbf.onnx no encontrado en el zip.")
        os.remove(zip_path)
    except Exception as e:
        print(f"\n[RECOG] Error descargando: {e}")
        print("[RECOG] Descarga manual en: https://github.com/deepinsight/insightface/releases/tag/v0.7")
        print("[RECOG] Extrae w600k_mbf.onnx en la carpeta models/")
        if os.path.exists(zip_path):
            os.remove(zip_path)


def _init_reconocimiento():
    global _ort_session, _ort_input, _recog_listo

    if _recog_listo:
        return
    _recog_listo = True

    model_path = os.path.join(_MODELS, "w600k_mbf.onnx")

    if not os.path.exists(model_path):
        _descargar_modelo(model_path)

    if not os.path.exists(model_path):
        print("[RECOG] Modelo no disponible. Instala onnxruntime y descarga el modelo.")
        return

    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        _ort_session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        _ort_input = _ort_session.get_inputs()[0].name
        print(f"[RECOG] MobileFaceNet listo | dims={_ort_session.get_outputs()[0].shape}")
    except ImportError:
        print("[RECOG] onnxruntime no instalado. Ejecuta: pip install onnxruntime")
    except Exception as e:
        print(f"[RECOG] Error cargando modelo: {e}")


def _alinear_cara(frame, face_row):
    """Alinea la cara a 112x112 con los 5 landmarks de YuNet."""
    src = np.array([
        [face_row[4],  face_row[5]],
        [face_row[6],  face_row[7]],
        [face_row[8],  face_row[9]],
        [face_row[10], face_row[11]],
        [face_row[12], face_row[13]],
    ], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(src, _TEMPLATE_112, method=cv2.LMEDS)
    if M is None:
        return None

    return cv2.warpAffine(frame, M, (112, 112),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _extraer_embedding(cara_112):
    """Extrae embedding L2-normalizado de 512 dims con MobileFaceNet."""
    if _ort_session is None:
        return None

    img = cv2.cvtColor(cara_112, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 127.5
    img = img.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 112, 112)

    out  = _ort_session.run(None, {_ort_input: img})[0][0]   # (512,)
    norm = np.linalg.norm(out)
    return (out / norm).astype(np.float32) if norm > 0 else out.astype(np.float32)


# =============================================================================
#  API PUBLICA
# =============================================================================

def _varianza_laplaciano(frame, x, y, w, h):
    """
    Calcula la varianza del Laplaciano del recorte de cara.
    Valor alto = textura real (cara). Valor bajo = fondo liso (pared IR).
    Usado por diagnostico.py para calibrar el filtro de deteccion.
    """
    recorte = frame[y:y+h, x:x+w]
    if recorte.size == 0:
        return 0.0
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY) if len(recorte.shape) == 3 else recorte
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def preprocesar_cara(gris_zona):
    return cv2.GaussianBlur(_clahe.apply(gris_zona), (3, 3), 0)


def extraer_caracteristicas(frame, haar_path=None, modo="auto", tipo_esperado=None):
    """
    Detecta cara, clasifica angulo y extrae embedding MobileFaceNet.
    Firma identica a la version LBP — compatible con interfaz.py sin cambios.
    """
    _init_detectores()
    _init_reconocimiento()

    caras = _detectar_caras(frame, tipo_esperado=tipo_esperado)
    if not caras:
        _bbox_smoother.update(None)   # resetear suavizador al perder la cara
        return None, None, None

    x, y, w, h, _ = caras[0]
    h_img, w_img   = frame.shape[:2]
    x1 = max(0, x);          y1 = max(0, y)
    x2 = min(w_img, x + w);  y2 = min(h_img, y + h)
    bbox_raw  = (x1, y1, x2 - x1, y2 - y1)
    bbox      = _bbox_smoother.update(bbox_raw)   # <-- suavizado EMA

    tipo = _clasificar_angulo(frame, bbox, frame.shape, tipo_esperado=None)

    embedding = None
    if _ultimo_face_yunet is not None:
        cara_alineada = _alinear_cara(frame, _ultimo_face_yunet)
        if cara_alineada is not None and cara_alineada.size > 0:
            embedding = _extraer_embedding(cara_alineada)

    if embedding is None:
        # Usar coordenadas crudas para el crop de reconocimiento
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
        recorte = frame[ry1:ry2, rx1:rx2]
        if recorte.size > 0:
            embedding = _extraer_embedding(cv2.resize(recorte, (112, 112)))

    return embedding, bbox, tipo


def distancia_coseno(v1, v2):
    """
    Distancia coseno entre vectores L2-normalizados. Rango [0, 2].
    < 0.40 misma persona  |  > 0.65 persona distinta
    """
    return float(1.0 - np.dot(v1, v2))


# Alias de compatibilidad — database.py importa distancia_chi2
distancia_chi2 = distancia_coseno


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
        cv2.putText(frame, etiquetas[tipo],
                    (x, y + h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    if texto:
        cv2.putText(frame, texto,
                    (x, max(14, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def guardar_rostro_recortado(frame, nombre="persona", carpeta_base="dataset", tipo_esperado=None):
    from datetime import datetime
    vector, bbox, tipo = extraer_caracteristicas(frame, tipo_esperado=tipo_esperado)
    if bbox is None:
        return None, None, None, None
    x, y, w, h   = bbox
    h_img, w_img = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
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