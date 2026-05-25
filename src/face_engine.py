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

  CAPAS DE OCLUSION (ESTA REVISION):
    REGLA: solo se detecta obstruccion en 4 zonas — mejillas, boca,
    nariz y ojos. La frente NO se chequea (cabello/fleco/pelo largo
    no debe disparar nada). La barba TAMPOCO debe disparar (el bigote
    y la barba tienen textura natural distinguible de tela uniforme).

    Capas activas:
      Capa 0 — Anti-spoofing ML (con auto-desactivacion)
      Capa 1 — HandLandmarker (manos)
      Capa 2 — IoU entre MP y YuNet (MP no encuentra cara coherente)
      Capa 3 — Boca sin piel (objeto cubriendo, threshold 0.08)
      Capa 4 — Lentes opacos (ambos ojos uniformes std<9)
      Capa 7 — POSICION + CONTENIDO de las 4 zonas (rediseñada)
      Capa 8 — Lower face skin/dark (red de seguridad extrema)

    Capas ELIMINADAS (causaban falsos positivos con TU cara):
      Capa 5 — Frente/gorra: TU PELO cae sobre frente → falso positivo constante
      Capa 6 — Boca std<10: TU BARBA da std 8-9 → falso positivo de "mascarilla"

MODELO REQUERIDO:
  Archivo: models/w600k_mbf.onnx
  Se descarga automaticamente al primer arranque (~16 MB).

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
_SCORE_MINIMO_CARA = 0.45
_LAPLACIAN_MIN     = 25.0

# -- Oclusion ---------------------------------------------------------------
# En pruebas reales la capa de posicion confundia barba, fleco y perfiles
# con "objeto en el rostro".  Estas constantes vuelven la oclusion
# conservadora: manos y objetos claros se bloquean, pero barba/fleco/perfil no.
_OCLUSION_DEBUG = False
_OCLUSION_SKIP_POS_EN_PERFIL = True
# Si es True, bloquea tapados claros/ropa en boca aunque MediaPipe siga
# encontrando landmarks aproximados sobre el objeto.
_OCLUSION_STRICT_MOUTH = True

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
    """

    def __init__(self, alpha: float = 0.35, jump_ratio: float = 0.70):
        self.alpha      = alpha
        self.jump_ratio = jump_ratio
        self._prev      = None

    def update(self, bbox):
        if bbox is None:
            self._prev = None
            return None

        new = tuple(float(v) for v in bbox)

        if self._prev is None:
            self._prev = new
            return bbox

        prev_cx = self._prev[0] + self._prev[2] / 2.0
        prev_cy = self._prev[1] + self._prev[3] / 2.0
        new_cx  = new[0] + new[2] / 2.0
        new_cy  = new[1] + new[3] / 2.0
        dist    = ((new_cx - prev_cx) ** 2 + (new_cy - prev_cy) ** 2) ** 0.5
        avg_face = (self._prev[2] + self._prev[3] + new[2] + new[3]) / 4.0

        if dist > avg_face * self.jump_ratio:
            self._prev = new
            return bbox

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

_bbox_smoother = _BboxSmoother(alpha=0.35)
_multiple_faces = False

# =============================================================================
#  DETECTOR YuNet + Haar
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

    min_face_side = max(60, min(w_img, h_img) // 8)

    detecciones = []
    for face in faces:
        x = int(face[0]); y = int(face[1])
        w = int(face[2]); h = int(face[3])
        score = float(face[14])
        x = max(0, x);  y = max(0, y)
        w = min(w, w_img - x);  h = min(h, h_img - y)
        if w < min_face_side or h < min_face_side:
            continue
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
#  CLASIFICACION DE ANGULO
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
    img = img.transpose(2, 0, 1)[np.newaxis]

    out  = _ort_session.run(None, {_ort_input: img})[0][0]
    norm = np.linalg.norm(out)
    return (out / norm).astype(np.float32) if norm > 0 else out.astype(np.float32)

# =============================================================================
#  API PUBLICA
# =============================================================================

def _varianza_laplaciano(frame, x, y, w, h):
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

    _multiple_faces = False
    if len(caras) > 1:
        c1 = caras[0]
        c2 = caras[1]
        area_1 = c1[2] * c1[3]
        area_2 = c2[2] * c2[3]
        score_2 = c2[4]

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
    bbox      = _bbox_smoother.update(bbox_raw)

    tipo = _clasificar_angulo(frame, bbox, frame.shape, tipo_esperado=None)

    if _multiple_faces:
        _bbox_smoother.update(None)
        return None, None, None

    embedding = None
    if _ultimo_face_yunet is not None:
        cara_alineada = _alinear_cara(frame, _ultimo_face_yunet)
        if cara_alineada is not None and cara_alineada.size > 0:
            embedding = _extraer_embedding(cara_alineada)

    if embedding is None:
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
        recorte = frame[ry1:ry2, rx1:rx2]
        if recorte.size > 0:
            embedding = _extraer_embedding(cv2.resize(recorte, (112, 112)))

    return embedding, bbox, tipo

# =============================================================================
#  DETECCION DE OCLUSION FACIAL
# =============================================================================

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
            output_facial_transformation_matrixes=True,
        )
        _mp_landmarker = _mp_vision.FaceLandmarker.create_from_options(opts)
        print("[MP] FaceLandmarker listo (con matriz 3D)")
    except Exception as e:
        print(f"[MP] Error inicializando FaceLandmarker: {e}")
        _mp_landmarker = None
    return _mp_landmarker


_mp_hand_landmarker    = None
_mp_hand_landmarker_ok = False
_MP_HAND_MODEL         = "hand_landmarker.task"



# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════

def _get_mp_hand_landmarker():
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
class _AntiSpoofDetector:
    DEFAULT_INPUT_SIZE = (80, 80)

    def __init__(self):
        self.session     = None
        self.input_name  = None
        self.input_size  = self.DEFAULT_INPUT_SIZE
        self.output_dims = None
        self._tried      = False

        self._calib_samples       = []
        self._CALIB_N             = 30
        self._CALIB_MIN_MEDIAN    = 0.65
        self._CALIB_MAX_STD       = 0.20
        self._auto_disabled       = False
        self._debug_first_n       = 5

    def _load(self):
        if self._tried:
            return
        self._tried = True

        disable_flag = os.path.join(_MODELS, "disable_anti_spoof.flag")
        if os.path.exists(disable_flag):
            print("[AntiSpoof] DESACTIVADO MANUALMENTE (flag file presente):", disable_flag)
            print("[AntiSpoof] Borra ese archivo para reactivar.")
            return

        model_path = os.path.join(_MODELS, "anti_spoof.onnx")
        if not os.path.exists(model_path):
            print("[AntiSpoof] modelo no encontrado en", model_path)
            return

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path, providers=['CPUExecutionProvider'])

            inp = self.session.get_inputs()[0]
            self.input_name = inp.name

            shape = inp.shape
            if len(shape) == 4:
                if isinstance(shape[2], int) and isinstance(shape[3], int):
                    self.input_size = (shape[3], shape[2])

            out_shape = self.session.get_outputs()[0].shape
            self.output_dims = out_shape[-1] if len(out_shape) >= 2 else 1

            print(f"[AntiSpoof] modelo cargado: input={self.input_size}, "
                  f"clases={self.output_dims}")
        except Exception as e:
            print(f"[AntiSpoof] error cargando modelo: {e}")
            self.session = None

    def is_available(self):
        self._load()
        return self.session is not None and not self._auto_disabled

    def predict_real(self, face_bgr):
        self._load()
        if self.session is None or self._auto_disabled:
            return None
        if face_bgr is None or face_bgr.size < 100:
            return None

        try:
            rgb       = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            resized   = cv2.resize(rgb, self.input_size)
            tensor    = resized.astype(np.float32) / 255.0
            tensor    = tensor.transpose(2, 0, 1)[np.newaxis, ...]

            outputs   = self.session.run(None, {self.input_name: tensor})
            logits    = np.array(outputs[0][0]).flatten()

            if logits.size == 1:
                prob = float(1.0 / (1.0 + np.exp(-logits[0])))
            else:
                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / np.sum(exp_l)

                if probs.size == 3:
                    prob = float(probs[1])
                elif probs.size == 2:
                    prob = float(probs[1])
                    if self._debug_first_n > 0:
                        print(f"[AntiSpoof] frame inicial: "
                              f"probs[0]={probs[0]:.3f} probs[1]={probs[1]:.3f} "
                              f"-> usado={prob:.3f}")
                        self._debug_first_n -= 1
                else:
                    prob = float(probs[-1])

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
                        print("           Sistema usara heuristicas (Capas 1-8).")
                        print("=" * 70)
                        return None
                    else:
                        print(f"[AntiSpoof] Calibracion OK: "
                              f"median={median_prob:.2f} std={std_prob:.2f}")

            return prob
        except Exception as e:
            print(f"[AntiSpoof] error en inferencia: {e}")
            return None


_anti_spoof_detector = None

def _get_anti_spoof():
    global _anti_spoof_detector
    if _anti_spoof_detector is None:
        _anti_spoof_detector = _AntiSpoofDetector()
    return _anti_spoof_detector


def detectar_oclusion(frame, bbox, tipo=None):
    """
    Detecta si el rostro esta obstruido — SOLO en 4 zonas:
        boca, nariz, ojos, mejillas.
    La FRENTE no se chequea (pelo/fleco no debe disparar).
    La BARBA no debe disparar (textura natural distinguible).

    Capas:
      Capa 0 — Anti-spoof ML (si disponible)
      Capa 1 — HandLandmarker (manos)
      Capa 8 — Lower face skin/dark (red seguridad)
      Capa 2 — IoU MP vs YuNet
      Capa 3 — Boca sin piel (objeto solido)
      Capa 4 — Lentes opacos
      Capa 7 — POSICION + CONTENIDO de las 4 zonas

    Capas ELIMINADAS:
      Capa 5 — frente/gorra (causaba falsos positivos con pelo largo)
      Capa 6 — boca std<10 (causaba falsos positivos con barba)
    """
    if bbox is None or frame is None:
        return False, ""

    h_img, w_img = frame.shape[:2]
    es_perfil = tipo in (TIPO_PERFIL_D, TIPO_PERFIL_I)

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

    # ── Helpers de contenido visual ─────────────────────────────────────────
    def _mask_piel(patch_bgr):
        """
        Máscara de piel más estricta que la versión anterior.
        La anterior podía contar tela blanca/gris como piel por saturación baja.
        Esta combina HSV + YCrCb y evita blancos/neutros brillantes.
        """
        if patch_bgr is None or patch_bgr.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        hsv   = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2YCrCb)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        y = ycrcb[:, :, 0]
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        # HSV: piel clara/morena suele tener tono rojo/naranja y saturación real.
        hsv_skin = (((h <= 28) | (h >= 160)) &
                    (s >= 16) & (s <= 230) &
                    (v >= 35))

        # YCrCb: ayuda con piel morena y cambios de luz.
        ycc_skin = ((cr >= 133) & (cr <= 183) &
                    (cb >= 72)  & (cb <= 138) &
                    (y  >= 28))

        # Evitar que papel/camisa blanca se cuente como piel.
        blanco_neutro = (s < 18) & (v > 145)

        mask = (hsv_skin | ycc_skin) & (~blanco_neutro)
        return (mask.astype(np.uint8) * 255)

    def _ratio_piel_mt(patch_bgr):
        if patch_bgr is None or patch_bgr.size == 0:
            return 0.0
        mask = _mask_piel(patch_bgr)
        return float(np.count_nonzero(mask)) / mask.size

    def _clip_roi(rx1, ry1, rx2, ry2):
        """ROI relativa al bbox YuNet/raw, con límites seguros."""
        x1p = max(0, int(fx1 + fw * rx1))
        y1p = max(0, int(fy1 + fh * ry1))
        x2p = min(w_img, int(fx1 + fw * rx2))
        y2p = min(h_img, int(fy1 + fh * ry2))
        if x2p <= x1p + 8 or y2p <= y1p + 8:
            return None
        patch = frame[y1p:y2p, x1p:x2p]
        return patch if patch.size > 120 else None

    def _metricas_roi(patch_bgr):
        """Métricas para diferenciar barba/sombra vs tela/papel/cubrebocas."""
        if patch_bgr is None or patch_bgr.size == 0:
            return {
                "skin": 0.0, "white": 0.0, "dark": 0.0,
                "sat_mean": 0.0, "val_mean": 0.0,
                "gray_std": 0.0, "lap": 0.0, "edge": 0.0,
            }
        hsv  = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        sat  = hsv[:, :, 1]
        val  = hsv[:, :, 2]
        size = max(1, sat.size)

        white_ratio = float(np.count_nonzero((sat < 55) & (val > 145))) / size
        dark_ratio  = float(np.count_nonzero(val < 60)) / size
        edges       = cv2.Canny(gray, 45, 120)
        edge_ratio  = float(np.count_nonzero(edges)) / size
        lap_var     = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return {
            "skin":     _ratio_piel_mt(patch_bgr),
            "white":    white_ratio,
            "dark":     dark_ratio,
            "sat_mean": float(np.mean(sat)),
            "val_mean": float(np.mean(val)),
            "gray_std": float(np.std(gray)),
            "lap":      lap_var,
            "edge":     edge_ratio,
        }

    def _bloqueo_boca_por_roi():
        """
        Bloqueo frontal de boca/nariz baja tapada.
        Usa ROIs fijos del bbox, no solo landmarks, porque MediaPipe puede
        inventar puntos de boca sobre una camisa/papel.
        """
        if es_perfil or not _OCLUSION_STRICT_MOUTH:
            return False, ""

        # Centro inferior: boca + bigote + parte alta del mentón.
        boca_roi = _clip_roi(0.22, 0.56, 0.78, 0.86)
        if boca_roi is None:
            return False, ""

        mb = _metricas_roi(boca_roi)

        # ROI de nariz/centro superior para comparar contra zona normalmente visible.
        nariz_roi = _clip_roi(0.30, 0.36, 0.70, 0.58)
        mn = _metricas_roi(nariz_roi) if nariz_roi is not None else None

        # Tela/papel blanco: mucha zona brillante con baja saturación y poca piel.
        blanco_tapando = (mb["white"] > 0.32 and
                          mb["skin"]  < 0.20 and
                          mb["val_mean"] > 115)

        # Cubrebocas/ropa oscura uniforme. La barba se salva porque suele tener
        # más bordes/textura; una tela oscura uniforme tiene menos edge/laplaciano.
        oscuro_uniforme = (mb["dark"] > 0.78 and
                           mb["skin"] < 0.12 and
                           mb["edge"] < 0.045 and
                           mb["lap"]  < 75.0)

        # Zona inferior casi sin piel comparada con nariz visible.
        contraste_con_nariz = False
        if mn is not None:
            contraste_con_nariz = (mn["skin"] > 0.12 and
                                   mb["skin"] < max(0.06, mn["skin"] * 0.38) and
                                   (mb["white"] > 0.22 or oscuro_uniforme))

        # Objeto claro/gris de baja saturación aunque no sea blanco puro.
        neutro_extenso = (mb["sat_mean"] < 48 and
                          mb["skin"] < 0.14 and
                          (mb["white"] > 0.24 or mb["gray_std"] < 38.0) and
                          mb["val_mean"] > 95)

        if blanco_tapando or oscuro_uniforme or contraste_con_nariz or neutro_extenso:
            if _OCLUSION_DEBUG:
                print("[OCL] Boca ROI: "
                      f"skin={mb['skin']:.2f} white={mb['white']:.2f} "
                      f"dark={mb['dark']:.2f} sat={mb['sat_mean']:.0f} "
                      f"val={mb['val_mean']:.0f} edge={mb['edge']:.3f} "
                      f"lap={mb['lap']:.1f}")
            return True, "mascara"

        return False, ""

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
            if real_prob is not None and real_prob < 0.30:
                print(f"[OCL] Capa 0: anti-spoof real_prob={real_prob:.3f} < 0.30")
                return True, "obstruccion"

    # ════════════════════════════════════════════════════════════════════════
    # CAPA 1: HandLandmarker
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
                pad_top  = max(10, fh // 8)
                pad_bot  = max(10, fh // 8)
                pad_side = max(10, fw // 6)
                hcx1 = fx1 - pad_side; hcy1 = fy1 - pad_top
                hcx2 = fx2 + pad_side; hcy2 = fy2 + pad_bot

                for hand in result.hand_landmarks:
                    dentro = 0
                    for lm in hand:
                        hx = lm.x * w_img
                        hy = lm.y * h_img
                        if hcx1 <= hx <= hcx2 and hcy1 <= hy <= hcy2:
                            dentro += 1
                    if dentro >= 5:
                        print(f"[OCL] Capa 1: mano confirmada ({dentro}/21 landmarks dentro)")
                        return True, "mano"
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════════════
    # CAPA 3A: Boca/nariz baja tapada por tela, papel o cubrebocas
    # ════════════════════════════════════════════════════════════════════════
    bloquea_boca, razon_boca = _bloqueo_boca_por_roi()
    if bloquea_boca:
        return True, razon_boca

    # ════════════════════════════════════════════════════════════════════════
    # CAPA 8: Red de seguridad — solo casos EXTREMOS
    # Mira SOLO la mitad inferior interior de la cara (no incluye frente).
    # ════════════════════════════════════════════════════════════════════════
    lf_y1 = fy1 + int(fh * 0.45)
    lf_y2 = fy1 + int(fh * 0.80)
    lf_x1 = fx1 + int(fw * 0.22)
    lf_x2 = fx2 - int(fw * 0.22)

    if lf_x2 > lf_x1 + 20 and lf_y2 > lf_y1 + 20:
        lower_face = frame[lf_y1:lf_y2, lf_x1:lf_x2]
        if lower_face.size > 300:
            m_lf = _metricas_roi(lower_face)
            ratio_lf = m_lf["skin"]
            # Solo bloquea objetos casi totalmente sin piel. Barba/sombra pasan
            # si tienen textura suficiente.
            th_skin_lf = 0.025 if es_perfil else 0.045
            if ratio_lf < th_skin_lf and (m_lf["white"] > 0.25 or
                                          (m_lf["dark"] > 0.86 and m_lf["edge"] < 0.04)):
                print(f"[OCL] Capa 8a: ratio_piel={ratio_lf:.3f} < {th_skin_lf:.2f} (objeto sin piel)")
                return True, "obstruccion"

            th_dark_lf = 0.97 if es_perfil else 0.94
            if (m_lf["dark"] > th_dark_lf and
                    m_lf["edge"] < 0.035 and m_lf["lap"] < 60.0):
                print(f"[OCL] Capa 8b: dark_ratio={m_lf['dark']:.3f} > {th_dark_lf:.2f} (zona oscura uniforme)")
                return True, "obstruccion"

    # IMPORTANTE: no salimos aunque anti-spoof esté activo.
    # Anti-spoof sirve para bloquear spoof/foto, pero si pasa, todavía debemos
    # revisar boca/ojos/nariz con las heurísticas de rasgos.

    # ════════════════════════════════════════════════════════════════════════
    # CAPAS 2-4 + 7: heuristicas con MediaPipe FaceLandmarker
    # NO incluye Capa 5 (frente) ni Capa 6 (boca std) — generaban falsos
    # positivos con pelo en frente y barba/bigote.
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
    iou_mp_yunet = inter_area / yunet_area
    # En perfiles el bbox de MediaPipe se mueve natural hacia un lado; no es oclusion.
    # Solo usamos este bloqueo en frontal y con margen relajado.
    if (not es_perfil) and iou_mp_yunet < 0.42:
        return True, "obstruccion"

    # ─── CAPA 3: Boca sin piel (objeto totalmente NO-piel) ──────────────────
    # Threshold MUY estricto (0.08) — solo dispara si hay CASI cero piel.
    # Barba moderada da 20-40% — pasa sin problemas.
    mouth_idx = [13, 14, 17, 0, 61, 291, 78, 308, 82, 312, 84, 314]
    mouth_xs  = [lms[i].x * w_img for i in mouth_idx if i < n]
    mouth_ys  = [lms[i].y * h_img for i in mouth_idx if i < n]
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
                m_mouth = _metricas_roi(patch)
                boca_tapada_clara = (m_mouth["white"] > 0.30 and
                                      m_mouth["skin"] < 0.20)
                boca_tapada_oscura = (m_mouth["dark"] > 0.82 and
                                       m_mouth["skin"] < 0.12 and
                                       m_mouth["edge"] < 0.05 and
                                       m_mouth["lap"] < 80.0)
                casi_sin_piel = (m_mouth["skin"] < 0.035 and
                                  (m_mouth["white"] > 0.18 or boca_tapada_oscura))
                if (not es_perfil) and (boca_tapada_clara or boca_tapada_oscura or casi_sin_piel):
                    if _OCLUSION_DEBUG:
                        print("[OCL] Capa 3 boca: "
                              f"skin={m_mouth['skin']:.2f} white={m_mouth['white']:.2f} "
                              f"dark={m_mouth['dark']:.2f} edge={m_mouth['edge']:.3f} "
                              f"lap={m_mouth['lap']:.1f}")
                    return True, "mascara"

    # ─── CAPA 4: Lentes OPACOS (no transparentes) ───────────────────────────
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

    # ════════════════════════════════════════════════════════════════════════
    # CAPA 7 — VERIFICACION POR POSICION + CONTENIDO (REDISEÑADA)
    # ════════════════════════════════════════════════════════════════════════
    # Para cada una de las 4 zonas (boca, nariz, ojos, mejillas) se evalua:
    #
    #   POSICION: ¿esta el rasgo donde DEBE estar en la cara?
    #     Tolerancia 0.20 (20% del bbox YuNet). Generoso para rotaciones leves.
    #
    #   CONTENIDO: ¿hay contenido coherente con ese rasgo?
    #     Thresholds MUY estrictos — solo dispara con obstruccion clara:
    #       boca:     skin ratio < 0.05  (mascara solida = sin piel)
    #                                     [barba con 0.15-0.30 PASA]
    #       nariz:    skin ratio < 0.10  (objeto sin piel)
    #                                     [piel oscura natural PASA]
    #       mejillas: skin ratio < 0.10
    #       ojos:     std < 3.0          (lentes opacos)
    #                                     [ojos oscuros naturales con std 5+ PASAN]
    #
    # Un rasgo FALLA si CUALQUIERA (posicion O contenido) falla.
    # Se necesitan 2 rasgos fallidos para confirmar obstruccion.
    #
    # Casos correctos:
    #   PELO en frente:    no afecta a los 4 rasgos -> 0 fallos -> NO dispara ✓
    #   BARBA:             posiciones OK, contenido OK (textura) -> 0 fallos -> NO dispara ✓
    #   OJOS oscuros:      posiciones OK, std > 5 -> 0 fallos -> NO dispara ✓
    #   PIEL morena:       ratio sigue > 0.10 con rangos HSV ampliados -> 0 fallos ✓
    #
    #   CAMISA en boca:    posiciones se desvian, MP confundido -> 2-4 fallos -> dispara ✓
    #   MASCARA real:      contenido boca sin piel -> 1-2 fallos -> dispara ✓
    #   LENTES opacos:     contenido ojos std < 3 -> 2 fallos -> dispara ✓
    #   MANO en cara:      Capa 1 dispara antes
    # ════════════════════════════════════════════════════════════════════════

    EXPECTED_POSITIONS = {
        "boca":     (0.50, 0.78),
        "nariz":    (0.50, 0.55),
        "ojo_izq":  (0.32, 0.40),
        "ojo_der":  (0.68, 0.40),
        "meji_izq": (0.27, 0.62),
        "meji_der": (0.73, 0.62),
    }

    FEATURE_INDICES = {
        "boca":     [13, 14, 17, 0, 61, 291, 78, 308],
        "nariz":    [1, 2, 4, 5, 19, 94, 125, 354],
        "ojo_izq":  [33, 133, 159, 145, 158, 153, 144, 163],
        "ojo_der":  [362, 263, 386, 374, 385, 380, 373, 390],
        "meji_izq": [205, 50, 142, 36, 100, 187],
        "meji_der": [425, 280, 371, 266, 329, 411],
    }

    # Si la cara esta de perfil, NO se valida posicion frontal.
    # Ese era el origen del falso "objeto en rostro": boca/nariz se desplazan
    # a x=0.75-0.90 de forma normal y la capa 7 los marcaba como fallidos.
    if es_perfil and _OCLUSION_SKIP_POS_EN_PERFIL:
        return False, ""

    # Tolerancia de posicion. Antes 0.20; era demasiado estricto.
    TOL_POS = 0.32

    # Threshold de contenido por rasgo: (tipo, valor minimo).
    # Bajos a proposito para aceptar barba, bigote, fleco, lentes comunes y piel morena.
    CONTENT_THRESH = {
        "boca":     ("skin", 0.045),
        "nariz":    ("skin", 0.05),
        "meji_izq": ("skin", 0.04),
        "meji_der": ("skin", 0.04),
        "ojo_izq":  ("std",  2.0),
        "ojo_der":  ("std",  2.0),
    }

    def _feature_center(indices):
        xs = [lms[i].x * w_img for i in indices if i < n]
        ys = [lms[i].y * h_img for i in indices if i < n]
        if len(xs) < 3:
            return None
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    def _patch(indices, pad=4):
        xs = [lms[i].x * w_img for i in indices if i < n]
        ys = [lms[i].y * h_img for i in indices if i < n]
        if len(xs) < 3:
            return None
        x1p = max(0, int(min(xs)) - pad); x2p = min(w_img, int(max(xs)) + pad)
        y1p = max(0, int(min(ys)) - pad); y2p = min(h_img, int(max(ys)) + pad)
        if x2p - x1p < 6 or y2p - y1p < 6:
            return None
        p = frame[y1p:y2p, x1p:x2p]
        return p if p.size > 60 else None

    fallos = []          # fallos reales de CONTENIDO
    pos_fallos = []      # solo diagnostico; no bloquean por si solos
    detalles = []

    for nombre, (exp_x, exp_y) in EXPECTED_POSITIONS.items():
        indices = FEATURE_INDICES[nombre]

        # Check 1: POSICION
        center = _feature_center(indices)
        pos_fail = False
        pos_detail = ""
        if center is not None:
            cx, cy = center
            rel_x = (cx - fx1) / fw if fw > 0 else 0.5
            rel_y = (cy - fy1) / fh if fh > 0 else 0.5
            if abs(rel_x - exp_x) > TOL_POS or abs(rel_y - exp_y) > TOL_POS:
                pos_fail = True
                pos_detail = f"pos=({rel_x:.2f},{rel_y:.2f})"

        # Check 2: CONTENIDO
        patch = _patch(indices, pad=4)
        cont_fail = False
        cont_detail = ""
        if patch is not None:
            check_type, threshold = CONTENT_THRESH[nombre]
            if check_type == "std":
                val = float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).std())
                if val < threshold:
                    cont_fail = True
                    cont_detail = f"std={val:.1f}"
            else:  # skin
                val = _ratio_piel_mt(patch)
                if val < threshold:
                    cont_fail = True
                    cont_detail = f"r={val:.2f}"

        # La posicion sola ya NO bloquea: en perfil, fleco, barba y mala luz
        # MediaPipe puede desplazar puntos aunque el rostro este descubierto.
        if cont_fail:
            razones = []
            if cont_fail: razones.append(cont_detail)
            if pos_fail:  razones.append(pos_detail)
            fallos.append(nombre)
            detalles.append(f"{nombre}({','.join(razones)})")
        elif pos_fail:
            pos_fallos.append(nombre)

    # ── Confirmacion conservadora de obstruccion ────────────────────────────
    # 2 fallos de contenido => bloqueo.
    # 1 fallo de contenido + 3 posiciones raras => bloqueo suave.
    # Solo posiciones raras => NO bloquea, solo diagnostico.
    if len(fallos) >= 2 or (len(fallos) >= 1 and len(pos_fallos) >= 3):
        if _OCLUSION_DEBUG:
            extra = f" | pos_solo={pos_fallos}" if pos_fallos else ""
            print(f"[OCL] Capa 7: contenido={len(fallos)} | {' | '.join(detalles)}{extra}")
        if "boca" in fallos:
            return True, "mascara"
        if "ojo_izq" in fallos and "ojo_der" in fallos:
            return True, "lentes"
        return True, "obstruccion"

    if _OCLUSION_DEBUG and len(pos_fallos) >= 2:
        print(f"[OCL] Capa 7 ignorada: solo posicion rara {pos_fallos} (barba/fleco/perfil/luz)")

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
        TIPO_FRONTAL:  (0, 212, 255),
        TIPO_PERFIL_D: (255, 165,   0),
        TIPO_PERFIL_I: (0,  165, 255),
    }
    c = colores_tipo.get(tipo, color)

    L      = max(28, w // 4)
    grosor = 3

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

    for p1, p2 in segmentos:
        cv2.line(frame, p1, p2, c, grosor, cv2.LINE_AA)

    etiquetas = {
        TIPO_FRONTAL:  "FRONTAL",
        TIPO_PERFIL_D: "PERFIL DER.",
        TIPO_PERFIL_I: "PERFIL IZQ.",
    }
    if tipo in etiquetas:
        etq_txt   = etiquetas[tipo]
        font      = cv2.FONT_HERSHEY_DUPLEX
        fscale    = 0.6
        fthick    = 2
        (tw, th), bl = cv2.getTextSize(etq_txt, font, fscale, fthick)
        tx = x + w // 2 - tw // 2
        ty = y + h + th + 10

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

        cv2.putText(frame, etq_txt, (tx + 2, ty + 2),
                    font, fscale, (0, 0, 0), fthick + 2, cv2.LINE_AA)
        cv2.putText(frame, etq_txt, (tx, ty),
                    font, fscale, c, fthick, cv2.LINE_AA)

    if texto:
        font2   = cv2.FONT_HERSHEY_DUPLEX
        fscale2 = 0.75
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

        bg = frame.copy()
        cv2.rectangle(bg, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.addWeighted(bg, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), c, 1, cv2.LINE_AA)

        for idx, line in enumerate(lines):
            (tw2, th2), _ = cv2.getTextSize(line, font2, fscale2, fthick2)
            tx2 = cx - tw2 // 2
            ty2 = by1 + (idx + 1) * line_h - 4
            cv2.putText(frame, line, (tx2 + 2, ty2 + 2),
                        font2, fscale2, (0, 0, 0), fthick2 + 2, cv2.LINE_AA)
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
