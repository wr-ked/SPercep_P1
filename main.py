# Perception. Practice 1. Camera calibration and 3D to 2D projection
# Authors: Hugo Villasan Atienza y Diego Lopez Salazar

import numpy as np
import cv2 as cv
import glob
import os
import time
import argparse
import open3d as o3d


# Parametros globales originales (Chessboard)
CHESSBOARD_FILAS = 9
CHESSBOARD_COLUMNAS = 6
TAM_CUADRADO = 0.02  # metros

CHARUCO_FILAS = 6
CHARUCO_COLUMNAS = 9
CHARUCO_TAM_CUADRADO = 0.026 # 26 mm en metros
CHARUCO_TAM_MARCADOR = 0.019  # 19 mm en metros
ARUCO_TAM_MARCADOR = 0.10
# Otros parametros
TERM_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
RMS_UMBRAL_ALTO = 0.8
CAPTURAS_REINTENTO = 20
MAX_REINTENTOS_RMS = 3
CALIBRATION_DIR_CHESSBOARD = "CalibrationImagesChessboard"
CALIBRATION_DIR_CHARUCO = "CalibrationImagesCharuco"

def build_object_points(centered=True):
    """Genera los puntos 3D del tablero sobre Z=0."""
    objp = np.zeros((CHESSBOARD_FILAS * CHESSBOARD_COLUMNAS, 3), np.float32)
    grid = np.mgrid[0:CHESSBOARD_COLUMNAS, 0:CHESSBOARD_FILAS].T.reshape(-1, 2)

    if centered:
        center = np.array([(CHESSBOARD_COLUMNAS - 1) / 2.0, (CHESSBOARD_FILAS - 1) / 2.0], dtype=np.float32)
        grid = grid - center

    objp[:, :2] = grid * TAM_CUADRADO
    return objp


def get_charuco_setup():
    """Crea y devuelve la configuracion de ChArUco compatible con API nueva y antigua."""
    if not hasattr(cv, "aruco"):
        print("OpenCV no incluye modulo aruco. Instala opencv-contrib-python")
        return None

    try:
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        board = cv.aruco.CharucoBoard(
            (CHARUCO_COLUMNAS, CHARUCO_FILAS),
            CHARUCO_TAM_CUADRADO,
            CHARUCO_TAM_MARCADOR,
            dictionary,
        )
        if hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(True)
        detector = cv.aruco.CharucoDetector(board)
        return {
            "old_api": False,
            "dictionary": dictionary,
            "board": board,
            "detector": detector,
            "detector_params": None,
        }
    except AttributeError:
        dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_250)
        board = cv.aruco.CharucoBoard_create(
            CHARUCO_COLUMNAS,
            CHARUCO_FILAS,
            CHARUCO_TAM_CUADRADO,
            CHARUCO_TAM_MARCADOR,
            dictionary,
        )
        detector_params = cv.aruco.DetectorParameters_create()
        return {
            "old_api": True,
            "dictionary": dictionary,
            "board": board,
            "detector": None,
            "detector_params": detector_params,
        }


def detect_charuco(gray, charuco_setup):
    """Detecta marcadores/esquinas ChArUco y devuelve corners+ids."""
    if charuco_setup is None:
        return None, None, None, None

    if not charuco_setup["old_api"]:
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_setup[
            "detector"
        ].detectBoard(gray)
        return charuco_corners, charuco_ids, marker_corners, marker_ids

    marker_corners, marker_ids, _ = cv.aruco.detectMarkers(
        gray,
        charuco_setup["dictionary"],
        parameters=charuco_setup["detector_params"],
    )

    if marker_ids is None or len(marker_ids) == 0:
        return None, None, marker_corners, marker_ids

    _, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        charuco_setup["board"],
    )
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def run_calibration(calibration_type):
    if calibration_type == "chessboard":
        return chessboard_calibration()
    if calibration_type == "aruco":
        return aruco_calibration()
    if calibration_type == "charuco":
        return charuco_calibration()

    print(f"Tipo de calibracion desconocido: {calibration_type}")
    return None, None, None, None, None


def calibration_image_captures(
    output_dir=CALIBRATION_DIR_CHESSBOARD,
    num_images=20,
    camera_index=0,
    interval_sec=1.5,
    pattern_type="chessboard",
):
    """Captura imagenes de calibracion y las guarda en output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # Contar imagenes existentes para no sobrescribir
    existing_images = []
    for ext in ("*.jpg", "*.JPG", "*.png", "*.jpeg"):
        existing_images.extend(glob.glob(os.path.join(output_dir, ext))) 
    next_index = len(existing_images) + 1 

    charuco_setup = None
    if pattern_type == "charuco":
        charuco_setup = get_charuco_setup()
        if charuco_setup is None:
            return []

    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        print("No se pudo abrir la camara para capturar imagenes de calibracion")
        return []

    print(f"Capturando {num_images} imagenes en la carpeta: {output_dir}")
    print(f"Patron seleccionado: {pattern_type}")
    print("Mueve el tablero por diferentes posiciones y orientaciones")
    print("Pulsa 'q' para cancelar")

    saved_images = []
    capture_count = 0
    last_capture_time = 0.0

    while capture_count < num_images:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer frame de la camara")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        preview = frame.copy()
        pattern_found = False

        if pattern_type == "chessboard":
            ret, corners = cv.findChessboardCorners(gray, (CHESSBOARD_COLUMNAS,CHESSBOARD_FILAS), None)
            if ret:
                pattern_found = True
                cv.drawChessboardCorners(preview, (CHESSBOARD_COLUMNAS,CHESSBOARD_FILAS), corners, ret)
        elif pattern_type == "charuco":
            charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco(
                gray, charuco_setup
            )

            if marker_ids is not None and len(marker_ids) > 0:
                cv.aruco.drawDetectedMarkers(preview, marker_corners, marker_ids)

            if (
                charuco_corners is not None
                and charuco_ids is not None
                and len(charuco_corners) >= 4
            ):
                pattern_found = True
                cv.aruco.drawDetectedCornersCharuco(
                    preview, charuco_corners, charuco_ids, (0, 255, 0)
                )
        else:
            print(f"Tipo de patron no soportado para captura: {pattern_type}")
            break

        if pattern_found:
            current_time = time.time()
            if current_time - last_capture_time >= interval_sec:
                capture_count += 1
                file_name = os.path.join(output_dir, f"calibration_{next_index:02d}.jpg")
                if cv.imwrite(file_name, frame):
                    saved_images.append(file_name)
                    last_capture_time = current_time
                    next_index += 1
                    print(f"Guardada {capture_count}/{num_images}: {file_name}")

        cv.putText(
            preview,
            f"Capturas: {capture_count}/{num_images}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv.imshow("Captura de calibracion", preview)

        if cv.waitKey(1) & 0xFF == ord("q"):
            print("Captura cancelada por usuario")
            break

    cap.release()
    cv.destroyAllWindows()

    print(f"Captura finalizada. Imagenes guardadas: {len(saved_images)}")
    return saved_images


def clear_calibration_images(output_dir=CALIBRATION_DIR_CHESSBOARD):
    """Elimina imagenes de calibracion de la carpeta indicada."""
    patterns = ("*.jpg", "*.JPG", "*.png", "*.jpeg")
    removed = 0

    for pattern in patterns:
        for file_path in glob.glob(os.path.join(output_dir, pattern)):
            try:
                os.remove(file_path)
                removed += 1
            except OSError as e:
                print(f"No se pudo borrar {file_path}: {e}")

    return removed


def chessboard_calibration():
    """Realiza la calibracion usando un patron de ajedrez y devuelve los parametros."""
    # Puntos 3D del patron (centrados para facilitar la proyeccion AR)
    objp = build_object_points(centered=True)

    objpoints = []
    imgpoints = []
    image_size = None

    # Usar carpeta dedicada para capturas de calibracion
    carpeta_imagenes = CALIBRATION_DIR_CHESSBOARD
    os.makedirs(carpeta_imagenes, exist_ok=True)

    extensions = ["*.jpg", "*.JPG", "*.png", "*.jpeg"]
    images = []
    for ext in extensions:
        ruta_busqueda = os.path.join(carpeta_imagenes, ext)
        images.extend(glob.glob(ruta_busqueda))

    if len(images) == 0:
        print(f"{carpeta_imagenes} esta vacia. Iniciando captura de 20 imagenes...")
        images = calibration_image_captures(
            output_dir=carpeta_imagenes,
            num_images=20,
            pattern_type="chessboard",
        )
        if len(images) == 0:
            return None, None, None, None, None

    print(f"Se usaran {len(images)} imagenes de: {carpeta_imagenes}")

    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print(f"Error leyendo {fname}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convertir a escala de grises para findChessboardCorners
        current_size = gray.shape[::-1] 

        # Proteccion basica contra imagenes de tamaños diferentes, que pueden causar errores en calibracion
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            print(f"Se omite {fname}: tamaño de imagen inconsistente")
            continue

        # Buscar el patron de ajedrez en la imagen
        ret, corners = cv.findChessboardCorners(gray, (CHESSBOARD_COLUMNAS,CHESSBOARD_FILAS), None)

        if ret:
            print(f"Patron encontrado en {fname}")
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERM_CRITERIA)
            imgpoints.append(corners2)

            cv.drawChessboardCorners(img, (CHESSBOARD_COLUMNAS,CHESSBOARD_FILAS), corners2, ret)
            # cv.imshow("Calibracion", img)
            # cv.waitKey(500)
        else:
            print(f"No se detecto el patron {CHESSBOARD_COLUMNAS}x{CHESSBOARD_FILAS} en {fname}")

    cv.destroyAllWindows()

    if len(objpoints) == 0 or len(imgpoints) == 0 or image_size is None:
        print("No se encontraron suficientes patrones validos para calibrar")
        return None, None, None, None, None

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    return ret, mtx, dist, rvecs, tvecs


def aruco_calibration():
    """Realiza la calibracion usando un unico marcador ArUco y devuelve los parametros."""
    print("\nIniciando calibracion con un unico marcador ArUco...")
    
    # Configuramos el detector (usamos diccionario 4x4 porque tu foto es de ese tipo)
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    detector_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, detector_params)

    # Definimos las 4 esquinas 3D de un solo marcador centrado en el origen (0,0,0)
    # Orden de OpenCV: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    L = ARUCO_TAM_MARCADOR / 2.0
    marker_3d = np.array([
        [-L,  L, 0],
        [ L,  L, 0],
        [ L, -L, 0],
        [-L, -L, 0]
    ], dtype=np.float32)

    all_obj_points = []
    all_img_points = []
    image_size = None

    # Pon la carpeta donde tengas las fotos de este marcador individual
    carpeta_imagenes = "Calibracion_Aruco_Diego" 
    os.makedirs(carpeta_imagenes, exist_ok=True)

    images = []
    for ext in ("*.jpg", "*.JPG", "*.png", "*.jpeg"):
        images.extend(glob.glob(os.path.join(carpeta_imagenes, ext)))

    if len(images) == 0:
        print(f"Error: La carpeta '{carpeta_imagenes}' esta vacia.")
        return None, None, None, None, None

    print(f"Se evaluaran {len(images)} imagenes de: {carpeta_imagenes}")

    for fname in images:
        img = cv.imread(fname)
        if img is None:
            continue
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if image_size is None:
            image_size = gray.shape[::-1]

        # Detectar el marcador
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # Usamos el primer marcador que encuentre (corners[0])
            # corners[0] tiene forma (1, 4, 2), extraemos las 4 esquinas 2D
            esquinas_2d = corners[0][0]
            
            all_obj_points.append(marker_3d)
            all_img_points.append(esquinas_2d)
            print(f"Marcador detectado en {fname}")
        else:
            print(f"Descartada {fname}: no se detecto el marcador")

    if len(all_obj_points) == 0:
        print("No se encontraron marcadores validos en las imagenes.")
        return None, None, None, None, None

    print("\nCalculando parametros de calibracion...")
    try:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            all_obj_points, all_img_points, image_size, None, None
        )
        return ret, mtx, dist, rvecs, tvecs
    except Exception as e:
        print(f"Error durante el calculo: {e}")
        return None, None, None, None, None
    
def charuco_calibration():
    """Realiza la calibracion usando un tablero ChArUco desde una carpeta y devuelve los parametros."""
    print("\nIniciando calibracion con ChArUco...")

    charuco_setup = get_charuco_setup()
    if charuco_setup is None:
        return None, None, None, None, None

    board = charuco_setup["board"]
    old_api = charuco_setup["old_api"]

    all_charuco_corners = []
    all_charuco_ids = []
    all_obj_points = []
    all_img_points = []
    image_size = None

    carpeta_imagenes = CALIBRATION_DIR_CHARUCO
    os.makedirs(carpeta_imagenes, exist_ok=True)

    images = []
    for ext in ("*.jpg", "*.JPG", "*.png", "*.jpeg"):
        images.extend(glob.glob(os.path.join(carpeta_imagenes, ext)))

    if len(images) == 0:
        print(f"{carpeta_imagenes} esta vacia. Iniciando captura de 20 imagenes...")
        images = calibration_image_captures(
            output_dir=carpeta_imagenes,
            num_images=20,
            pattern_type="charuco",
        )
        if len(images) == 0:
            return None, None, None, None, None

    print(f"Se evaluaran {len(images)} imagenes de: {carpeta_imagenes}")

    for fname in images:
        img = cv.imread(fname)
        if img is None:
            continue
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            print(f"Se omite {fname}: tamaño de imagen inconsistente")
            continue

        charuco_corners, charuco_ids, _, marker_ids = detect_charuco(gray, charuco_setup)

        if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
            if marker_ids is None or len(marker_ids) == 0:
                print(f"Descartada {fname}: no se detecto ningun marcador")
            else:
                print(f"Descartada {fname}: no hay suficientes esquinas ChArUco")
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

        if not old_api:
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
            if (
                obj_points is None
                or img_points is None
                or len(obj_points) < 4
                or len(img_points) < 4
            ):
                print(f"Descartada {fname}: no se pudieron emparejar puntos 2D/3D")
                continue
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)

        print(f"ChArUco valido en {fname} ({len(charuco_corners)} esquinas)")

    if len(all_charuco_corners) == 0:
        print("No se encontraron tableros validos en ninguna de las imagenes proporcionadas.")
        return None, None, None, None, None

    # 3. Calibrar la camara
    print("\nCalculando parametros de calibracion...")
    try:
        if not old_api:
            if len(all_obj_points) == 0 or len(all_img_points) == 0 or image_size is None:
                print("No hay suficientes puntos validos para calibrar con ChArUco")
                return None, None, None, None, None
            # Calibracion estandar de OpenCV usando los puntos extraidos del tablero
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                all_obj_points, all_img_points, image_size, None, None
            )
        else:
            if image_size is None:
                print("No se pudo determinar el tamaño de imagen para calibrar")
                return None, None, None, None, None
            # Funcion deprecada en versiones nuevas, pero necesaria en las antiguas
            ret, mtx, dist, rvecs, tvecs = cv.aruco.calibrateCameraCharuco(
                charucoCorners=all_charuco_corners,
                charucoIds=all_charuco_ids,
                board=board,
                imageSize=image_size,
                cameraMatrix=None,
                distCoeffs=None
            )
        return ret, mtx, dist, rvecs, tvecs
        
    except Exception as e:
        print(f"Error durante el calculo matematico: {e}")
        return None, None, None, None, None


def load_pcd_model(ruta_pcd, escala=1.0):
    """Carga una nube de puntos PCD como array float32 para OpenCV."""
    if o3d is None:
        print("open3d no esta instalado; se omite carga de modelo PCD")
        return None

    if not os.path.exists(ruta_pcd):
        print(f"No existe el archivo PCD: {ruta_pcd}")
        return None

    try:
        pcd = o3d.io.read_point_cloud(ruta_pcd)
        puntos_3d = np.asarray(pcd.points)
        if puntos_3d.size == 0:
            print(f"El archivo PCD no contiene puntos: {ruta_pcd}")
            return None

        puntos_3d = np.float32(puntos_3d * escala)
        print(f"Modelo cargado correctamente con {len(puntos_3d)} puntos")
        return puntos_3d
    except Exception as e:
        print(f"No se pudo cargar el modelo PCD ({ruta_pcd}): {e}")
        return None


def live_projection(mtx, dist, objp, puntos_modelo, calibration_type="chessboard"):
    charuco_setup = None
    if calibration_type == "charuco":
        charuco_setup = get_charuco_setup()
        if charuco_setup is None:
            print("No se pudo inicializar ChArUco para la proyeccion en vivo")
            return

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la camara")
        return

    print("Presiona 'q' para cerrar la proyeccion en vivo")

    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    detector = cv.aruco.ArucoDetector(dictionary, cv.aruco.DetectorParameters())

    # Las mismas 4 esquinas 3D que usamos para calibrar
    L = ARUCO_TAM_MARCADOR / 2.0
    marker_3d = np.array([
        [-L,  L, 0],
        [ L,  L, 0],
        [ L, -L, 0],
        [-L, -L, 0]
    ], dtype=np.float32)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detectar el marcador
        corners, ids, rejected = detector.detectMarkers(gray)

        if debe_buscar:
            if calibration_type == "charuco":
                board = charuco_setup["board"]
                charuco_corners, charuco_ids, _, _ = detect_charuco(
                    gray, charuco_setup
                )

                if (
                    charuco_corners is not None
                    and charuco_ids is not None
                    and len(charuco_corners) >= 4
                ):
                    patron_detectado = True

                    ret_pnp = False
                    rvec, tvec = None, None

                    try:
                        obj_points, img_points = board.matchImagePoints(
                            charuco_corners, charuco_ids
                        )
                        if (
                            obj_points is not None
                            and img_points is not None
                            and len(obj_points) >= 4
                        ):
                            ret_pnp, rvec, tvec = cv.solvePnP(
                                obj_points, img_points, mtx, dist
                            )
                    except Exception:
                        ret_pnp = False

                    if not ret_pnp and hasattr(cv.aruco, "estimatePoseCharucoBoard"):
                        try:
                            ret_pnp, rvec, tvec = cv.aruco.estimatePoseCharucoBoard(
                                charuco_corners,
                                charuco_ids,
                                board,
                                mtx,
                                dist,
                                None,
                                None,
                            )
                        except Exception:
                            ret_pnp = False

                    if ret_pnp and puntos_modelo is not None:
                        imgpts, _ = cv.projectPoints(puntos_modelo, rvec, tvec, mtx, dist)
                        for pt in imgpts:
                            x, y = int(pt[0][0]), int(pt[0][1])
                            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                                cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
                else:
                    patron_detectado = False
                    cv.putText(
                        frame,
                        "Buscando patron ChArUco",
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
            else:
                # CALIB_CB_FAST_CHECK hace una pasada rápida y aborta si no ve un tablero
                flags_opt = (
                    cv.CALIB_CB_ADAPTIVE_THRESH
                    + cv.CALIB_CB_FAST_CHECK
                    + cv.CALIB_CB_NORMALIZE_IMAGE
                )
                ret_pattern, corners = cv.findChessboardCorners(
                    gray, (CHESSBOARD_COLUMNAS,CHESSBOARD_FILAS), flags=flags_opt
                )

                if ret_pattern:
                    patron_detectado = True
                    corners2 = cv.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), TERM_CRITERIA
                    )
                    ret_pnp, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

                    if ret_pnp and puntos_modelo is not None:
                        imgpts, _ = cv.projectPoints(puntos_modelo, rvec, tvec, mtx, dist)
                        for pt in imgpts:
                            x, y = int(pt[0][0]), int(pt[0][1])
                            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                                cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
                else:
                    patron_detectado = False
                    cv.putText(
                        frame,
                        "Buscando patron",
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
        else:
            # En los frames saltados, mantenemos el feedback visual
            if calibration_type == "charuco":
                label = "Buscando patron ChArUco"
            else:
                label = "Buscando patron"
            cv.putText(
                frame,
                label,
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv.imshow("Realidad Aumentada ArUco", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

def main(calibration_type="chessboard", show_intrinsic=False, show_extrinsic=False):
    print("=== PRACTICA 1: CALIBRACION Y PROYECCION ===")

    print(f"\nRealizando calibracion con tipo: {calibration_type.upper()}...")
    
    ret, mtx, dist, rvecs, tvecs = run_calibration(calibration_type)

    # Si el error de reproyeccion es alto, intentamos capturar nuevas imagenes y recalibrar
    reintentos = 0
    while mtx is not None and ret > RMS_UMBRAL_ALTO and reintentos < MAX_REINTENTOS_RMS:
        reintentos += 1
        print(
            f"\nAviso: error de reproyeccion alto ({ret:.6f} > {RMS_UMBRAL_ALTO:.3f})."
        )
        print(f"Reintento automatico {reintentos}/{MAX_REINTENTOS_RMS}")

        if calibration_type == "chessboard":
            calibration_dir = CALIBRATION_DIR_CHESSBOARD
        elif calibration_type == "charuco":
            calibration_dir = CALIBRATION_DIR_CHARUCO
        else:
            print(
                "El reintento automatico de captura no esta implementado para este tipo de calibracion"
            )
            break

        borradas = clear_calibration_images(output_dir=calibration_dir)
        print(f"Se borraron {borradas} imagenes de {calibration_dir}")

        nuevas = calibration_image_captures(
            output_dir=calibration_dir,
            num_images=CAPTURAS_REINTENTO,
            pattern_type=calibration_type,
        )

        if len(nuevas) == 0:
            print("No se capturaron nuevas imagenes. No se puede repetir la calibracion")
            break

        print("Recalculando calibracion con las nuevas imagenes...")
        ret, mtx, dist, rvecs, tvecs = run_calibration(calibration_type)

    if mtx is not None and ret > RMS_UMBRAL_ALTO:
        print(
            f"\nAdvertencia: el error de reproyeccion sigue alto ({ret:.6f}) tras los reintentos"
        )

    # Mostrar resultados de calibracion
    if mtx is not None:
        print("\nCalibracion exitosa")
        
        # Mostrar parametros intrinsecos si se solicita
        if show_intrinsic:
            print("\n=== MATRIZ DE PARAMETROS INTRINSECOS (M) ===")
            print("Matriz de camara:")
            print(mtx)
            print(f"\nDistancia focal fx: {mtx[0,0]:.2f}")
            print(f"Distancia focal fy: {mtx[1,1]:.2f}")
            print(f"Centro optico cx: {mtx[0,2]:.2f}")
            print(f"Centro optico cy: {mtx[1,2]:.2f}")

            print("\nCoeficientes de distorsion:")
            print(dist.ravel())

        # Mostrar parametros extrinsecos si se solicita
        if show_extrinsic:
            print("\n=== PARAMETROS EXTRINSECOS PARA CADA IMAGEN ===")
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                print(f"\n--- IMAGEN {i+1} ---")

                R, _ = cv.Rodrigues(rvec)

                print("Matriz de rotacion R:")
                print(R)

                print("Vector de traslacion T:")
                print(tvec.ravel())

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec.ravel()

                print("Matriz de transformacion completa T (4x4):")
                print(T)

        print("\n=== RESUMEN DE CALIBRACION ===")
        print(f"Error de reproyeccion RMS: {ret:.6f}")
        print(f"Imagenes utilizadas: {len(rvecs)}")
        print("\nPreparando proyeccion en vivo...")
        if calibration_type == "chessboard":
            objp = build_object_points(centered=True)
        else:
            objp = None
        escala_modelo = 0.05 * TAM_CUADRADO
        puntos_pcd = load_pcd_model("ninetales_voxelizado.pcd", escala=escala_modelo)
        live_projection(
            mtx,
            dist,
            objp,
            puntos_pcd,
            calibration_type=calibration_type,
        )
    else:
        print("No se puede continuar sin calibracion valida")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibracion de camara y proyeccion 3D a 2D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ejemplos de uso:\n  python main.py -c chessboard                       # Calibrar con patrón de ajedrez\n  python main.py -c aruco                            # Calibrar con marcadores ArUco\n  python main.py -c charuco                          # Calibrar con patrón ChArUco\n  python main.py -c chessboard -I                    # Con matriz intrínseca\n  python main.py -c chessboard -E                    # Con matrices extrínsecas\n  python main.py -c chessboard -I -E                 # Con ambas matrices"
    )
    parser.add_argument(
        "-c", "--calibration-type",
        type=str,
        choices=["chessboard", "aruco", "charuco"],
        required=True,
        help="Tipo de patron de calibracion a utilizar (obligatorio)"
    )
    parser.add_argument(
        "-I", "--show-intrinsic",
        action="store_true",
        help="Mostrar matriz de parametros intrinsecos"
    )
    parser.add_argument(
        "-E", "--show-extrinsic",
        action="store_true",
        help="Mostrar matriz de parametros extrinsecos para cada imagen"
    )
    
    args = parser.parse_args()
    main(
        calibration_type=args.calibration_type,
        show_intrinsic=args.show_intrinsic,
        show_extrinsic=args.show_extrinsic
    )
