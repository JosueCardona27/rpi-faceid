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

    def __init__(self, alpha: float = 0.35, jump_ratio: float = 0.70):
        self.alpha      = alpha
        self.jump_ratio = jump_ratio
        self._prev      = None   # (x, y, w, h) en flotante

    def update(self, bbox):
        """
        Recibe bbox crudo (x, y, w, h) o None si no hay cara.
        Devuelve bbox suavizado como tupla de enteros, o None.

        Si la nueva cara esta muy lejos de la actual (salto mayor al
        70% del tamaño de la cara), se resetea instantaneamente en
        lugar de interpolar — evita el efecto loco con varias personas.
        """
        if bbox is None:
            self._prev = None
            return None

        new = tuple(float(v) for v in bbox)

        if self._prev is None:
            self._prev = new
            return bbox

        # Detectar salto grande (probable cambio de persona)
        prev_cx = self._prev[0] + self._prev[2] / 2.0
        prev_cy = self._prev[1] + self._prev[3] / 2.0
        new_cx  = new[0] + new[2] / 2.0
        new_cy  = new[1] + new[3] / 2.0
        dist    = ((new_cx - prev_cx) ** 2 + (new_cy - prev_cy) ** 2) ** 0.5
        avg_face = (self._prev[2] + self._prev[3] + new[2] + new[3]) / 4.0

        if dist > avg_face * self.jump_ratio:
            # Salto = otra persona mas cercana; resetear sin interpolar
            self._prev = new
            return bbox

        # Suavizado EMA normal
        a  = self.alpha
        sx = a * new[0] + (1 - a) * self._prev[0]
        sy = a * new[1] + (1 - a) * self._prev[1]
        sw = a * new[2] + (1 - a) * self._prev[2]
        sh = a * new[3] + (1 - a) * self._prev[3]
        self._prev = (sx, sy, sw, sh)

        return (int(round(sx)), int(round(sy)),
                int(round(sw)), int(round(sh)))

    def reset(self):
        self._prev = None

# Instancia global
_bbox_smoother = _BboxSmoother(alpha=0.35)

# Señal para interfaz.py: True cuando hay varias caras de tamaño similar
_multiple_faces = False

# =============================================================================
#  DETECTOR YuNet + Haar  (identico a version anterior)
# =============================================================================

_yunet        = None
_haar_frontal = None
_haar_perfil  = None
_haar_eye     = None
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
                    # score_threshold subido de 0.30 -> 0.55:
                    # 0.30 dejaba pasar reflejos, esquinas oscuras, patrones de tela
                    # 0.55 sigue siendo permisivo para caras reales pero filtra ruido.
                    score_threshold=0.55,
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

    # Haar eye cascade (para deteccion de oclusion)
    global _haar_eye
    _haar_eye = None
    rutas_e = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        rutas_e.append(cv2.data.haarcascades + "haarcascade_eye.xml")
        rutas_e.append(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    for ruta in rutas_e:
        if os.path.exists(ruta):
            clf = cv2.CascadeClassifier(ruta)
            if not clf.empty():
                _haar_eye = clf
                print(f"[DET] Haar eye: {ruta}")
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

    # ── Filtro de tamano minimo absoluto ─────────────────────────────────────
    # Una cara real para escaneo necesita ocupar al menos ~12% del lado mas
    # corto del frame. Caras de 15x15 px son SIEMPRE falsos positivos
    # (reflejos, patrones de tela, esquinas de objetos).
    #
    # En 640x480 -> minimo 58x58 px aproximadamente.
    # En 800x600 -> minimo 72x72 px.
    min_face_side = max(60, min(w_img, h_img) // 8)

    detecciones = []
    for face in faces:
        x = int(face[0]); y = int(face[1])
        w = int(face[2]); h = int(face[3])
        score = float(face[14])
        x = max(0, x);  y = max(0, y)
        w = min(w, w_img - x);  h = min(h, h_img - y)
        # Filtro 1: tamano absoluto razonable (no detecciones diminutas)
        if w < min_face_side or h < min_face_side:
            continue
        # Filtro 2: proporciones razonables de cara (no objetos alargados)
        # Caras humanas: ratio w/h tipicamente entre 0.55 y 1.20.
        aspect = w / float(h)
        if aspect < 0.55 or aspect > 1.20:
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

    global _multiple_faces

    caras = _detectar_caras(frame, tipo_esperado=tipo_esperado)
    if not caras:
        _bbox_smoother.update(None)
        _multiple_faces = False
        return None, None, None

    # ── Deteccion de multiples personas (logica robusta) ─────────────────────
    # Una "segunda cara" valida debe cumplir TRES condiciones simultaneamente:
    #   1. Tamano similar a la principal (>= 60% del area)
    #   2. Score de confianza alto (>= 0.70) — descarta detecciones marginales
    #      en el fondo (TVs, micros, patrones de tela)
    #   3. Separacion fisica razonable (centros separados al menos por la mitad
    #      del ancho de la cara) — descarta double-detection sobre el mismo rostro
    #
    # Si las tres se cumplen → realmente hay dos personas y mostramos el banner.
    _multiple_faces = False
    if len(caras) > 1:
        c1 = caras[0]   # (x, y, w, h, score)
        c2 = caras[1]
        area_1 = c1[2] * c1[3]
        area_2 = c2[2] * c2[3]
        score_2 = c2[4]

        # Centros de cada cara
        cx1 = c1[0] + c1[2] / 2.0
        cy1 = c1[1] + c1[3] / 2.0
        cx2 = c2[0] + c2[2] / 2.0
        cy2 = c2[1] + c2[3] / 2.0
        dist_centros = ((cx2 - cx1)**2 + (cy2 - cy1)**2) ** 0.5
        ancho_1 = c1[2]

        cond_tamano       = area_2 > area_1 * 0.60
        cond_confianza    = score_2 >= 0.70
        cond_separacion   = dist_centros > ancho_1 * 0.5

        if cond_tamano and cond_confianza and cond_separacion:
            _multiple_faces = True
            print(f"[DET] Multi-rostro detectado: "
                  f"c1=({c1[2]}x{c1[3]} s={c1[4]:.2f}), "
                  f"c2=({c2[2]}x{c2[3]} s={c2[4]:.2f}), "
                  f"dist={dist_centros:.0f}px")

    x, y, w, h, _ = caras[0]
    h_img, w_img   = frame.shape[:2]
    x1 = max(0, x);          y1 = max(0, y)
    x2 = min(w_img, x + w);  y2 = min(h_img, y + h)
    bbox_raw  = (x1, y1, x2 - x1, y2 - y1)
    bbox      = _bbox_smoother.update(bbox_raw)   # <-- suavizado EMA

    tipo = _clasificar_angulo(frame, bbox, frame.shape, tipo_esperado=None)

    if _multiple_faces:
        # Varias personas de tamaño similar: ocultar todo, solo mostrar mensaje.
        # NO devolvemos bbox para que NO se dibuje ningun recuadro y se evite
        # cualquier oscilacion visual entre los rostros detectados.
        _bbox_smoother.update(None)
        return None, None, None

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

# =============================================================================
#  DETECCION DE OCLUSION FACIAL
# =============================================================================

# ── MediaPipe Face Mesh (singleton, carga una sola vez) ───────────────────────
# ── MediaPipe FaceLandmarker (deteccion de cara con malla densa) ─────────────
_mp_landmarker    = None
_mp_landmarker_ok = False
_MP_FACE_MODEL    = "face_landmarker.task"


def _descargar_mp_model(dest, modelo="face"):
    """Descarga modelos de MediaPipe (face o hand) si no existen."""
    import urllib.request
    urls = {
        "face": "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "hand": "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    }
    url = urls.get(modelo)
    if url is None:
        return False

    nombre = "FaceLandmarker" if modelo == "face" else "HandLandmarker"
    print(f"[MP] Descargando modelo {nombre}...")
    try:
        urllib.request.urlretrieve(url, dest)
        if os.path.getsize(dest) < 1000:
            os.remove(dest)
            print(f"[MP] Descarga fallida (archivo muy pequeno)")
            return False
        print(f"[MP] Modelo guardado en {dest}")
        return True
    except Exception as e:
        print(f"[MP] Error descargando modelo: {e}")
        return False


def _get_mp_landmarker():
    """Inicializa MediaPipe FaceLandmarker la primera vez."""
    global _mp_landmarker, _mp_landmarker_ok
    if _mp_landmarker_ok:
        return _mp_landmarker
    _mp_landmarker_ok = True

    model_path = os.path.join(_MODELS, _MP_FACE_MODEL)
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        if not _descargar_mp_model(model_path, "face"):
            return None

    try:
        from mediapipe.tasks import python as _mp_python
        from mediapipe.tasks.python import vision as _mp_vision

        base_opts = _mp_python.BaseOptions(model_asset_path=model_path)
        opts = _mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,  # para detectar yaw 3D
        )
        _mp_landmarker = _mp_vision.FaceLandmarker.create_from_options(opts)
        print("[MP] FaceLandmarker listo (con matriz 3D)")
    except Exception as e:
        print(f"[MP] Error inicializando FaceLandmarker: {e}")
        _mp_landmarker = None
    return _mp_landmarker


# ── MediaPipe HandLandmarker (deteccion directa de manos) ────────────────────
_mp_hand_landmarker    = None
_mp_hand_landmarker_ok = False
_MP_HAND_MODEL         = "hand_landmarker.task"


def _get_mp_hand_landmarker():
    """Inicializa MediaPipe HandLandmarker la primera vez."""
    global _mp_hand_landmarker, _mp_hand_landmarker_ok
    if _mp_hand_landmarker_ok:
        return _mp_hand_landmarker
    _mp_hand_landmarker_ok = True

    model_path = os.path.join(_MODELS, _MP_HAND_MODEL)
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        if not _descargar_mp_model(model_path, "hand"):
            return None

    try:
        from mediapipe.tasks import python as _mp_python
        from mediapipe.tasks.python import vision as _mp_vision

        base_opts = _mp_python.BaseOptions(model_asset_path=model_path)
        opts = _mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_hands=2,
            # Subido de 0.15 -> 0.50:
            # 0.15 disparaba con objetos del fondo (microfono, cables, sabanas,
            # esquinas oscuras de TVs). 0.50 sigue detectando manos reales
            # (incluso en poses dificiles) pero filtra fantasmas.
            min_hand_detection_confidence=0.50,
            min_hand_presence_confidence=0.50,
        )
        _mp_hand_landmarker = _mp_vision.HandLandmarker.create_from_options(opts)
        print("[MP] HandLandmarker listo")
    except Exception as e:
        print(f"[MP] Error inicializando HandLandmarker: {e}")
        _mp_hand_landmarker = None
    return _mp_hand_landmarker


# ═════════════════════════════════════════════════════════════════════════════
# ANTI-SPOOFING ML (Silent-Face MiniFASNet)
# ═════════════════════════════════════════════════════════════════════════════
# Modelo open-source que clasifica si un rostro es REAL (persona en vivo) o
# SPOOF (foto, pantalla, mascara, objeto cubriendo el rostro).
#
# Para activar:
#   1. Descarga un modelo Silent-Face en formato ONNX (~2-5 MB).
#      Opcion A: https://huggingface.co/datasets/Wuvin/Unique3D/blob/main/
#                ckpt/onnx_face_anti_spoofing/2.7_80x80_MiniFASNetV2.onnx
#      Opcion B: https://github.com/hairymax/Face-AntiSpoofing
#                (carpeta saved_models)
#   2. Renombra el archivo a 'anti_spoof.onnx'.
#   3. Colocalo en la carpeta models/ del proyecto.
#
# Si el archivo no esta presente, el sistema cae en heuristicas (8 capas).

class _AntiSpoofDetector:
    """
    Detector anti-spoofing usando un modelo ONNX (MiniFASNet o similar).

    Carga perezosa: el modelo se carga la primera vez que se llama predict_real.
    Si el archivo models/anti_spoof.onnx no existe, is_available() devuelve
    False y el sistema usa heuristicas como respaldo.
    """

    DEFAULT_INPUT_SIZE = (80, 80)

    def __init__(self):
        self.session     = None
        self.input_name  = None
        self.input_size  = self.DEFAULT_INPUT_SIZE
        self.output_dims = None      # cantidad de clases del modelo
        self._tried      = False     # solo intentar cargar una vez

        # Calibracion automatica con DOS criterios de auto-desactivacion:
        #   A. Mediana baja: si el modelo no logra dar probabilidad alta de forma
        #      consistente para rostros reales, esta sesgado.
        #   B. Alta varianza: si el modelo da 0.91 en un frame y 0.19 en el
        #      siguiente para la MISMA cara, es ruido, no senal.
        # En cualquier caso → se desactiva y se usa Capa 8 + heuristicas.
        self._calib_samples       = []
        self._CALIB_N             = 30       # frames para decidir
        self._CALIB_MIN_MEDIAN    = 0.65     # mediana minima aceptable
        self._CALIB_MAX_STD       = 0.20     # desviacion maxima (estabilidad)
        self._auto_disabled       = False
        self._debug_first_n       = 5        # imprimir detalle de N primeros

    def _load(self):
        if self._tried:
            return
        self._tried = True

        # ── Escape manual: archivo "disable_anti_spoof.flag" en models/ ──
        # Si el archivo existe, el modelo no se carga ni siquiera. El sistema
        # corre directamente con heuristicas. Util cuando se sabe que el
        # modelo es poco fiable para esta camara/usuario.
        disable_flag = os.path.join(_MODELS, "disable_anti_spoof.flag")
        if os.path.exists(disable_flag):
            print("[AntiSpoof] DESACTIVADO MANUALMENTE (flag file presente):", disable_flag)
            print("[AntiSpoof] Borra ese archivo para reactivar.")
            print("[AntiSpoof] Sistema usara heuristicas (Capas 2-8) tuneadas.")
            return

        model_path = os.path.join(_MODELS, "anti_spoof.onnx")
        if not os.path.exists(model_path):
            print("[AntiSpoof] modelo no encontrado en", model_path)
            print("[AntiSpoof] sistema usara heuristicas como respaldo")
            print("[AntiSpoof] para activar, descarga un modelo MiniFASNet ONNX")
            print("[AntiSpoof] y colocalo en:", model_path)
            return

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path, providers=['CPUExecutionProvider'])

            inp = self.session.get_inputs()[0]
            self.input_name = inp.name

            # Auto-detectar tamaño de entrada del modelo (HxW)
            shape = inp.shape
            if len(shape) == 4:
                # Formato NCHW: [N, C, H, W]
                if isinstance(shape[2], int) and isinstance(shape[3], int):
                    self.input_size = (shape[3], shape[2])  # (W, H) para cv2.resize

            # Auto-detectar cantidad de clases de salida
            out_shape = self.session.get_outputs()[0].shape
            self.output_dims = out_shape[-1] if len(out_shape) >= 2 else 1

            print(f"[AntiSpoof] modelo cargado: input={self.input_size}, "
                  f"clases={self.output_dims}")
        except Exception as e:
            print(f"[AntiSpoof] error cargando modelo: {e}")
            self.session = None

    def is_available(self):
        self._load()
        # Si la calibracion automatica desactivo el modelo, ya no esta disponible.
        return self.session is not None and not self._auto_disabled

    def predict_real(self, face_bgr):
        """
        Devuelve probabilidad [0, 1] de que el rostro sea REAL.
        None si el modelo no esta disponible o falla la inferencia.

        Calibracion automatica:
          Si los primeros 30 frames con rostro consistentemente dan prob < 0.40,
          el modelo se desactiva automaticamente (sesgo del modelo).
        """
        self._load()
        if self.session is None or self._auto_disabled:
            return None
        if face_bgr is None or face_bgr.size < 100:
            return None

        try:
            # Preprocesamiento: BGR -> RGB, resize, normalizar a [0,1]
            rgb       = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            resized   = cv2.resize(rgb, self.input_size)
            tensor    = resized.astype(np.float32) / 255.0
            tensor    = tensor.transpose(2, 0, 1)[np.newaxis, ...]  # HWC->CHW + batch

            outputs   = self.session.run(None, {self.input_name: tensor})
            logits    = np.array(outputs[0][0]).flatten()

            # Manejar salidas: 1 valor (sigmoid), 2 clases (binario), 3 clases (Silent-Face)
            if logits.size == 1:
                # Sigmoid: salida directa es probabilidad de real
                prob = float(1.0 / (1.0 + np.exp(-logits[0])))
            else:
                # Softmax para multiclase
                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / np.sum(exp_l)

                if probs.size == 3:
                    # Silent-Face estandar: [spoof_2d, real, spoof_3d]
                    prob = float(probs[1])
                elif probs.size == 2:
                    # AntiSpoofing_bin_1.5_128 (hairymax):
                    # Verificacion empirica con TU usuario muestra que la clase
                    # de mayor probabilidad para rostros reales es la clase 1,
                    # no la 0 como originalmente se asumio. Por eso usamos probs[1].
                    #
                    # En este modelo binario:
                    #   probs[0] = probabilidad de SPOOF
                    #   probs[1] = probabilidad de REAL  ← devolvemos esta
                    prob = float(probs[1])

                    # Log detallado de los primeros frames para confirmacion visual
                    if self._debug_first_n > 0:
                        print(f"[AntiSpoof] frame inicial: "
                              f"probs[0]={probs[0]:.3f} probs[1]={probs[1]:.3f} "
                              f"-> usado={prob:.3f}")
                        self._debug_first_n -= 1
                else:
                    # Asumir que la clase "real" es la de mayor indice
                    prob = float(probs[-1])

            # ── Calibracion automatica de sesgo del modelo ────────────────
            # DOS criterios de fallo (cualquiera dispara desactivacion):
            #   A. Mediana < 0.65 → modelo da poca probabilidad a rostros reales
            #   B. Desviacion std > 0.20 → modelo inestable, frame a frame da
            #      respuestas contradictorias (0.91 luego 0.19 luego 0.60).
            if len(self._calib_samples) < self._CALIB_N:
                self._calib_samples.append(prob)
                if len(self._calib_samples) == self._CALIB_N:
                    samples = np.array(self._calib_samples)
                    median_prob = float(np.median(samples))
                    std_prob    = float(np.std(samples))
                    min_prob    = float(np.min(samples))
                    max_prob    = float(np.max(samples))

                    fail_median = median_prob < self._CALIB_MIN_MEDIAN
                    fail_std    = std_prob    > self._CALIB_MAX_STD

                    if fail_median or fail_std:
                        self._auto_disabled = True
                        razon = []
                        if fail_median:
                            razon.append(f"mediana={median_prob:.2f}<{self._CALIB_MIN_MEDIAN}")
                        if fail_std:
                            razon.append(f"std={std_prob:.2f}>{self._CALIB_MAX_STD} (inestable)")
                        print("=" * 70)
                        print("[AntiSpoof] AUTO-DESACTIVADO. Razon:", ", ".join(razon))
                        print(f"           Muestras: min={min_prob:.2f} max={max_prob:.2f} "
                              f"median={median_prob:.2f} std={std_prob:.2f}")
                        print("           El modelo es poco fiable para esta "
                              "camara/iluminacion/usuario.")
                        print("           Sistema usara heuristicas (Capas 2-8) tuneadas")
                        print("           para tolerar barba, fleco, piel oscura.")
                        print("=" * 70)
                        return None
                    else:
                        print(f"[AntiSpoof] Calibracion OK: "
                              f"median={median_prob:.2f} std={std_prob:.2f}")

            return prob
        except Exception as e:
            print(f"[AntiSpoof] error en inferencia: {e}")
            return None


# Instancia global, carga perezosa
_anti_spoof_detector = None


def _get_anti_spoof():
    global _anti_spoof_detector
    if _anti_spoof_detector is None:
        _anti_spoof_detector = _AntiSpoofDetector()
    return _anti_spoof_detector


def detectar_oclusion(frame, bbox, tipo=None):
    """
    Detecta si el rostro esta obstruido durante el registro/verificacion.

    ESTRUCTURA (corto-circuito):

    Siempre activas (tolerantes a barba, lentes transparentes, cabello, piel oscura):
      Capa 0  Anti-spoofing ML         [si modelo disponible]
      Capa 1  HandLandmarker (manos)
      Capa 8  Skin/dark ratio cara inf. — red de seguridad robusta

    Solo si NO hay anti-spoof (modo heuristico):
      Capa 2  IoU MP vs YuNet
      Capa 3  Skin ratio en boca
      Capa 4  Contraste en ojos (solo lentes MUY oscuros, std<9)
      Capa 5  Skin ratio en frente (umbral bajo para cabello/fleco)
      Capa 6  Varianza en boca (umbral bajo para barba)
      Capa 7  Verificacion holistica de rasgos por pose

    Tolerancias ajustadas para condiciones reales:
      - Barba: umbrales de std y ratio de piel mas bajos
      - Lentes transparentes: threshold std ojos 12 -> 9 (solo lentes muy oscuros)
      - Cabello/fleco en frente: threshold forehead 0.25 -> 0.12
      - Piel morena: rango HSV ampliado para incluir tonos oscuros
      - Iluminacion directa: rango V de 50 bajado a 30
    """
    # ─── Validaciones iniciales ─────────────────────────────────────────────
    if bbox is None or frame is None:
        return False, ""

    h_img, w_img = frame.shape[:2]

    if _ultimo_face_yunet is not None:
        fr = _ultimo_face_yunet
        x_raw = int(fr[0]); y_raw = int(fr[1])
        w_raw = int(fr[2]); h_raw = int(fr[3])
    else:
        x_raw, y_raw, w_raw, h_raw = bbox

    fx1 = max(0, x_raw);  fy1 = max(0, y_raw)
    fx2 = min(w_img, x_raw + w_raw);  fy2 = min(h_img, y_raw + h_raw)
    fw  = fx2 - fx1;  fh = fy2 - fy1
    if fw < 30 or fh < 30:
        return False, ""

    yunet_area = float(fw * fh)

    # ── Helper de deteccion de piel multi-tono ───────────────────────────────
    # Cubre piel clara, media, morena y oscura con iluminacion variable.
    # Incluye tres rangos HSV:
    #   m1: piel clara/media (H=0-30, V>=30)
    #   m2: reflejos rosados/rojos (H=140-179)
    #   m3: piel oscura/morena con baja luminosidad (V=20-90, S moderada)
    def _mask_piel(patch_bgr):
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, np.array([0,    8, 25]),  np.array([30,  255, 255]))
        m2  = cv2.inRange(hsv, np.array([140,  8, 25]),  np.array([179, 255, 255]))
        m3  = cv2.inRange(hsv, np.array([5,   20, 15]),  np.array([25,  200, 90]))
        return cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)

    def _ratio_piel_mt(patch_bgr):
        """Ratio de piel multi-tono sobre el total de pixeles del parche."""
        if patch_bgr is None or patch_bgr.size == 0:
            return 0.0
        mask = _mask_piel(patch_bgr)
        return float(np.count_nonzero(mask)) / mask.size

    # ════════════════════════════════════════════════════════════════════════
    # CAPA 0: Anti-spoofing ML
    # ════════════════════════════════════════════════════════════════════════
    anti_spoof = _get_anti_spoof()
    anti_spoof_active = anti_spoof.is_available()
    if anti_spoof_active:
        pad = max(8, min(fw, fh) // 6)
        as_x1 = max(0, fx1 - pad);  as_y1 = max(0, fy1 - pad)
        as_x2 = min(w_img, fx2 + pad);  as_y2 = min(h_img, fy2 + pad)
        face_crop = frame[as_y1:as_y2, as_x1:as_x2]
        if face_crop.size > 200:
            real_prob = anti_spoof.predict_real(face_crop)
            # Threshold permisivo (0.30) para tolerar variabilidad por:
            # - vello facial denso (barba, bigote)
            # - iluminacion directa o lateral
            # - tonos de piel oscura (el modelo se entreno con dataset sesgado)
            # - lentes con reflejos
            # Solo dispara cuando el modelo esta MUY seguro que NO es real.
            if real_prob is not None and real_prob < 0.30:
                print(f"[OCL] Capa 0: anti-spoof real_prob={real_prob:.3f} < 0.30")
                return True, "obstruccion"

    # ════════════════════════════════════════════════════════════════════════
    # CAPA 1: HandLandmarker
    # Condiciones estrictas para considerar que una mano OCLUYE el rostro:
    #   - Padding moderado (mano debe estar SOBRE la cara, no debajo)
    #   - Minimo 5 landmarks dentro del area (no 2) — landmarks aislados
    #     suelen ser artefactos. Una mano real tiene los 21 landmarks juntos.
    # ════════════════════════════════════════════════════════════════════════
    hand_lmk = _get_mp_hand_landmarker()
    frame_rgb = None
    if hand_lmk is not None:
        try:
            import mediapipe as _mp
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=frame_rgb)
            result    = hand_lmk.detect(mp_img)
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                # Padding ajustado:
                # - pad_top: la mano puede entrar por arriba (frente)
                # - pad_bot reducido de fh//3 a fh//8: no extender hasta el cuello/pecho
                #   donde el microfono u objetos generaban "manos" fantasma
                # - pad_side: similar
                pad_top  = max(10, fh // 8)
                pad_bot  = max(10, fh // 8)   # reducido de fh//3
                pad_side = max(10, fw // 6)   # reducido de fw//5
                hcx1 = fx1 - pad_side; hcy1 = fy1 - pad_top
                hcx2 = fx2 + pad_side; hcy2 = fy2 + pad_bot

                for hand in result.hand_landmarks:
                    dentro = 0
                    for lm in hand:
                        hx = lm.x * w_img
                        hy = lm.y * h_img
                        if hcx1 <= hx <= hcx2 and hcy1 <= hy <= hcy2:
                            dentro += 1
                    # Subido de 2 -> 5 landmarks:
                    # Una mano real ocluyendo la cara mete al menos 5 landmarks
                    # (yemas + nudillos) dentro del bbox. 2 landmarks son
                    # tipicamente ruido o un dedo lejano sin tapar la cara.
                    if dentro >= 5:
                        print(f"[OCL] Capa 1: mano confirmada ({dentro}/21 landmarks dentro)")
                        return True, "mano"
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════════════
    # CAPA 8: Red de seguridad robusta (siempre activa)
    # Solo dispara en casos EXTREMOS — confianza principal va al anti-spoof.
    # Tolerante con: barba, bigote, piel oscura, iluminacion directa, sombras.
    #
    # Region INTERIOR del rostro (no incluye cuello ni borde):
    #   Vertical:   45% -> 80% del bbox YuNet (antes 45%-95%)
    #               Excluye el cuello/menton bajo donde caian sombras.
    #   Horizontal: 22% -> 78% del bbox YuNet (antes 15%-85%)
    #               Mas estrecho para no incluir cabello lateral / borde.
    # ════════════════════════════════════════════════════════════════════════
    lf_y1 = fy1 + int(fh * 0.45)
    lf_y2 = fy1 + int(fh * 0.80)   # reducido de 0.95 -> 0.80 (no cuello)
    lf_x1 = fx1 + int(fw * 0.22)   # subido de 0.15 -> 0.22 (no borde)
    lf_x2 = fx2 - int(fw * 0.22)

    if lf_x2 > lf_x1 + 20 and lf_y2 > lf_y1 + 20:
        lower_face = frame[lf_y1:lf_y2, lf_x1:lf_x2]
        if lower_face.size > 300:
            ratio_lf = _ratio_piel_mt(lower_face)
            # Cara normal con barba/piel oscura: 30-90% piel
            # Telefono / mascara solida / libro: <12%
            # Threshold MUY permisivo: solo dispara con casi cero piel.
            if ratio_lf < 0.12:
                print(f"[OCL] Capa 8a: ratio_piel={ratio_lf:.3f} < 0.12 (objeto sin piel)")
                return True, "obstruccion"
            # Objeto MUY oscuro (V<55) dominando la zona inferior.
            # Threshold 0.78 (antes 0.65) para no disparar con:
            # - sombras laterales por luz direccional
            # - bigote/barba densa
            # - piel oscura en sombra
            hsv_lf     = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
            dark_ratio = float(np.count_nonzero(hsv_lf[:,:,2] < 55)) / hsv_lf[:,:,2].size
            if dark_ratio > 0.78:
                print(f"[OCL] Capa 8b: dark_ratio={dark_ratio:.3f} > 0.78 (zona muy oscura)")
                return True, "obstruccion"

    # Con anti-spoof + manos + Capa 8 ya verificados, parar aqui
    if anti_spoof_active:
        return False, ""

    # ════════════════════════════════════════════════════════════════════════
    # CAPAS 2-7: Heuristicas (solo sin anti-spoof). Con thresholds ajustados:
    #   - Barba: thresholds de std y ratio mas bajos
    #   - Lentes transparentes: std ojos de 12 -> 9 (solo lentes MUY oscuros)
    #   - Cabello/fleco: threshold frente de 0.25 -> 0.12
    #   - Piel morena: helper _ratio_piel_mt en lugar de rangos fijos
    # ════════════════════════════════════════════════════════════════════════
    try:
        import mediapipe as _mp
        if frame_rgb is None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_full = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=frame_rgb)
    except Exception:
        return False, ""

    lmk = _get_mp_landmarker()
    if lmk is None:
        return False, ""

    try:
        result = lmk.detect(mp_full)
    except Exception:
        return False, ""

    if not result.face_landmarks:
        return True, "obstruccion"

    lms = result.face_landmarks[0]
    n   = len(lms)

    mp_xs = np.array([lm.x * w_img for lm in lms])
    mp_ys = np.array([lm.y * h_img for lm in lms])
    mp_x1 = float(np.min(mp_xs)); mp_x2 = float(np.max(mp_xs))
    mp_y1 = float(np.min(mp_ys)); mp_y2 = float(np.max(mp_ys))
    mp_w  = mp_x2 - mp_x1; mp_h = mp_y2 - mp_y1; mp_area = mp_w * mp_h

    if mp_area < 100 or mp_w < 40 or mp_h < 40:
        return False, ""

    # ─── CAPA 2: IoU MP vs YuNet ────────────────────────────────────────────
    inter_x1 = max(fx1, mp_x1); inter_y1 = max(fy1, mp_y1)
    inter_x2 = min(fx2, mp_x2); inter_y2 = min(fy2, mp_y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area / yunet_area < 0.55:
        return True, "obstruccion"

    # ─── CAPA 3: Skin ratio en boca (obj NO-piel) ───────────────────────────
    # Threshold 0.08 (antes 0.10): barba moderada da 20-40% con rango ampliado
    mouth_idx = [13, 14, 17, 0, 61, 291, 78, 308, 82, 312, 84, 314]
    mouth_xs  = [lms[i].x * w_img for i in mouth_idx if i < n]
    mouth_ys  = [lms[i].y * h_img for i in mouth_idx if i < n]
    pad_x = pad_y = 0
    if len(mouth_xs) >= 6:
        bx1 = int(min(mouth_xs)); bx2 = int(max(mouth_xs))
        by1 = int(min(mouth_ys)); by2 = int(max(mouth_ys))
        pad_x = max(8, (bx2 - bx1) // 3)
        pad_y = max(8, (by2 - by1) // 3)
        bx1 = max(0, bx1 - pad_x); bx2 = min(w_img, bx2 + pad_x)
        by1 = max(0, by1 - pad_y); by2 = min(h_img, by2 + pad_y)
        if bx2 > bx1 + 10 and by2 > by1 + 10:
            patch = frame[by1:by2, bx1:bx2]
            if patch.size > 100:
                if _ratio_piel_mt(patch) < 0.08:
                    return True, "mascara"

    # ─── CAPA 4: Lentes OSCUROS (solo lentes opacos, no transparentes) ───────
    # Threshold std: 9 (antes 12). Lentes transparentes tienen std > 9 por el iris.
    # Solo dispara si AMBOS ojos son muy uniformes y oscuros (std < 9).
    def _ojo_muy_uniforme(eye_idx):
        xs = [lms[i].x * w_img for i in eye_idx if i < n]
        ys = [lms[i].y * h_img for i in eye_idx if i < n]
        if len(xs) < 4: return False
        pad = max(4, (int(max(xs)) - int(min(xs))) // 5)
        ex1 = max(0, int(min(xs)) - pad); ex2 = min(w_img, int(max(xs)) + pad)
        ey1 = max(0, int(min(ys)) - pad); ey2 = min(h_img, int(max(ys)) + pad)
        if ex2 - ex1 < 8 or ey2 - ey1 < 5: return False
        ep = frame[ey1:ey2, ex1:ex2]
        if ep.size < 40: return False
        return float(cv2.cvtColor(ep, cv2.COLOR_BGR2GRAY).std()) < 9.0

    if _ojo_muy_uniforme([33,133,159,145,158,153,144,163]) and        _ojo_muy_uniforme([362,263,386,374,385,380,373,390]):
        return True, "lentes"

    # ─── CAPA 5: Gorra (frente debe ser piel) ───────────────────────────────
    # Threshold ratio: 0.08 (antes 0.12). Fleco/cabello en frente NO debe disparar.
    # Solo dispara si la frente esta MUY cubierta (gorra/visera baja casi a las cejas).
    if all(i < n for i in (10, 9, 67, 297)):
        fh_x1 = int(min(lms[67].x * w_img, lms[297].x * w_img))
        fh_x2 = int(max(lms[67].x * w_img, lms[297].x * w_img))
        fh_y1 = int(lms[10].y * h_img)
        fh_y2 = int(lms[9].y  * h_img)
        if fh_x2 - fh_x1 > 20 and fh_y2 - fh_y1 > 8:
            forehead = frame[fh_y1:fh_y2, fh_x1:fh_x2]
            if forehead.size > 100:
                rf = _ratio_piel_mt(forehead)
                if rf < 0.08:
                    print(f"[OCL] Capa 5: forehead skin={rf:.3f} < 0.08 (gorra)")
                    return True, "gorra"

    # ─── CAPA 6: Varianza en boca (mascarilla uniforme) ─────────────────────
    # Threshold std: 10 (antes 12). Barba densa con poca luz da std ~10-15.
    if len(mouth_xs) >= 6:
        bx1m = max(0, int(min(mouth_xs)) - pad_x)
        bx2m = min(w_img, int(max(mouth_xs)) + pad_x)
        by1m = max(0, int(min(mouth_ys)) - pad_y)
        by2m = min(h_img, int(max(mouth_ys)) + pad_y)
        if bx2m > bx1m + 10 and by2m > by1m + 10:
            pm = frame[by1m:by2m, bx1m:bx2m]
            if pm.size > 100:
                std_m = float(cv2.cvtColor(pm, cv2.COLOR_BGR2GRAY).std())
                if std_m < 10.0:
                    print(f"[OCL] Capa 6: mouth std={std_m:.2f} < 10.0 (mascarilla)")
                    return True, "mascara"

    # ─── CAPA 7: Verificacion holistica por pose (CON VOTACION) ──────────────
    # CAMBIO IMPORTANTE: ya no falla con un solo rasgo mal — usa VOTACION.
    # Cada rasgo (boca, nariz, ojo izq/der, mejilla izq/der) suma un voto.
    # FRONTAL: necesita 3+ "fallos" sobre 6 rasgos para disparar.
    # PERFIL:  necesita 2+ "fallos" sobre 4 rasgos esenciales para disparar.
    #
    # Esto tolera situaciones reales como:
    #   - Bigote denso (puede bajar std de boca)
    #   - Piel oscura con sombra en una mejilla (mejilla baja ratio)
    #   - Un solo ojo en sombra natural
    # Solo dispara cuando VARIOS rasgos coinciden = obstruccion real.
    #
    # Thresholds bajados para tolerar barba/piel oscura/sombras:
    #   boca std:  > 10   (antes 14)
    #   nariz:     > 0.18 (antes 0.28) - piel oscura con sombra nasal
    #   ojos std:  > 7    (antes 9)
    #   mejillas:  > 0.20 (antes 0.35) - sombras laterales por luz direccional
    def _patch(indices, pad=4):
        xs = [lms[i].x * w_img for i in indices if i < n]
        ys = [lms[i].y * h_img for i in indices if i < n]
        if len(xs) < 3: return None
        x1 = max(0, int(min(xs)) - pad); x2 = min(w_img, int(max(xs)) + pad)
        y1 = max(0, int(min(ys)) - pad); y2 = min(h_img, int(max(ys)) + pad)
        if x2 - x1 < 6 or y2 - y1 < 6: return None
        p = frame[y1:y2, x1:x2]
        return p if p.size > 60 else None

    def _std_v(patch):
        if patch is None: return 0.0
        return float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).std())

    boca_std    = _std_v(_patch([13,14,17,0,61,291,78,308], 6))
    nariz_r     = _ratio_piel_mt(_patch([1,2,4,5,19,94,125,354], 3))
    ojo_izq_std = _std_v(_patch([33,133,159,145,158,153,144,163], 4))
    ojo_der_std = _std_v(_patch([362,263,386,374,385,380,373,390], 4))
    meji_izq_r  = _ratio_piel_mt(_patch([205,50,142,36,100,187], 4))
    meji_der_r  = _ratio_piel_mt(_patch([425,280,371,266,329,411], 4))

    # Cada rasgo da TRUE si esta visible (no obstruido)
    boca_ok     = boca_std    > 10.0
    nariz_ok    = nariz_r     > 0.18
    ojo_izq_ok  = ojo_izq_std > 7.0
    ojo_der_ok  = ojo_der_std > 7.0
    meji_izq_ok = meji_izq_r  > 0.20
    meji_der_ok = meji_der_r  > 0.20

    # Contar fallos
    fallos = sum(not v for v in (boca_ok, nariz_ok, ojo_izq_ok, ojo_der_ok,
                                  meji_izq_ok, meji_der_ok))

    if tipo == TIPO_FRONTAL:
        # FRONTAL: necesita 3 o mas rasgos fallidos para disparar
        umbral_fallos = 3
    else:
        # PERFIL: solo 4 rasgos visibles, necesita 2 o mas fallos
        umbral_fallos = 2

    if fallos >= umbral_fallos:
        print(f"[OCL] Capa 7: {fallos} rasgos fallidos (>={umbral_fallos}) | "
              f"boca_std={boca_std:.1f} nariz_r={nariz_r:.2f} "
              f"ojoI_std={ojo_izq_std:.1f} ojoD_std={ojo_der_std:.1f} "
              f"mejI_r={meji_izq_r:.2f} mejD_r={meji_der_r:.2f}")
        # Razon = la mas comun
        if not boca_ok:
            return True, "mascara"
        if not (ojo_izq_ok or ojo_der_ok):
            return True, "lentes"
        return True, "obstruccion"

    return False, ""



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

    colores_tipo = {
        TIPO_FRONTAL:  (0, 212, 255),   # cian
        TIPO_PERFIL_D: (255, 165,   0), # naranja
        TIPO_PERFIL_I: (0,  165, 255),  # azul
    }
    c = colores_tipo.get(tipo, color)

    # ── Longitud de esquinas y grosor ────────────────────────────────────────
    L      = max(28, w // 4)   # antes: max(18, w//4) — esquinas más largas
    grosor = 3                  # antes: 2 — línea más gruesa

    # ── Sombra de las esquinas (desplazada 2 px, color negro) ────────────────
    segmentos = [
        ((x,     y),     (x + L,     y)),       ((x,     y),     (x,     y + L)),
        ((x + w, y),     (x + w - L, y)),       ((x + w, y),     (x + w, y + L)),
        ((x,     y + h), (x + L,     y + h)),   ((x,     y + h), (x,     y + h - L)),
        ((x + w, y + h), (x + w - L, y + h)),   ((x + w, y + h), (x + w, y + h - L)),
    ]
    for p1, p2 in segmentos:
        cv2.line(frame,
                 (p1[0] + 2, p1[1] + 2),
                 (p2[0] + 2, p2[1] + 2),
                 (0, 0, 0), grosor + 2, cv2.LINE_AA)

    # ── Esquinas del marco ───────────────────────────────────────────────────
    for p1, p2 in segmentos:
        cv2.line(frame, p1, p2, c, grosor, cv2.LINE_AA)

    # ── Etiqueta de tipo de pose (parte inferior del marco) ──────────────────
    etiquetas = {
        TIPO_FRONTAL:  "FRONTAL",
        TIPO_PERFIL_D: "PERFIL DER.",
        TIPO_PERFIL_I: "PERFIL IZQ.",
    }
    if tipo in etiquetas:
        etq_txt   = etiquetas[tipo]
        font      = cv2.FONT_HERSHEY_DUPLEX   # antes SIMPLEX — más legible
        fscale    = 0.6                        # antes: 0.45
        fthick    = 2                          # antes: 1
        (tw, th), bl = cv2.getTextSize(etq_txt, font, fscale, fthick)
        tx = x + w // 2 - tw // 2
        ty = y + h + th + 10

        # Fondo píldora semitransparente
        pad = 5
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (tx - pad,      ty - th - pad),
                      (tx + tw + pad, ty + bl + pad - 2),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
        cv2.rectangle(frame,
                      (tx - pad,      ty - th - pad),
                      (tx + tw + pad, ty + bl + pad - 2),
                      c, 1, cv2.LINE_AA)

        # Sombra + texto
        cv2.putText(frame, etq_txt, (tx + 2, ty + 2),
                    font, fscale, (0, 0, 0), fthick + 2, cv2.LINE_AA)
        cv2.putText(frame, etq_txt, (tx, ty),
                    font, fscale, c, fthick, cv2.LINE_AA)

    # ── Texto principal (sobre el marco, parte superior) ─────────────────────
    if texto:
        font2   = cv2.FONT_HERSHEY_DUPLEX   # antes SIMPLEX
        fscale2 = 0.75                       # antes: 0.6
        fthick2 = 2
        lines   = texto.split("\n")
        line_h  = 28
        cx      = x + w // 2

        max_tw = max(
            cv2.getTextSize(ln, font2, fscale2, fthick2)[0][0]
            for ln in lines
        )
        total_h = len(lines) * line_h
        bx1 = cx - max_tw // 2 - 8
        bx2 = cx + max_tw // 2 + 8
        by1 = max(0, y - total_h - 12)
        by2 = max(total_h + 4, y - 4)

        # Fondo semitransparente negro con borde coloreado
        bg = frame.copy()
        cv2.rectangle(bg, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.addWeighted(bg, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), c, 1, cv2.LINE_AA)

        for idx, line in enumerate(lines):
            (tw2, th2), _ = cv2.getTextSize(line, font2, fscale2, fthick2)
            tx2 = cx - tw2 // 2
            ty2 = by1 + (idx + 1) * line_h - 4
            # Sombra
            cv2.putText(frame, line, (tx2 + 2, ty2 + 2),
                        font2, fscale2, (0, 0, 0), fthick2 + 2, cv2.LINE_AA)
            # Texto
            cv2.putText(frame, line, (tx2, ty2),
                        font2, fscale2, color, fthick2, cv2.LINE_AA)

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