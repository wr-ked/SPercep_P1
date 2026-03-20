# Perception. Practice 1. Camera calibration and 3D to 2D projection
# Authors: Hugo Villasan Atienza y Diego Lopez Salazar

import numpy as np
import cv2 as cv
import glob
import os
import time
import argparse
import open3d as o3d


# Parametros globales
FILAS = 9
COLUMNAS = 6
TAM_CUADRADO = 0.02  # metros
TERM_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
RMS_UMBRAL_ALTO = 0.8
CAPTURAS_REINTENTO = 20
MAX_REINTENTOS_RMS = 3


def build_object_points(centered=True):
    """Genera los puntos 3D del tablero sobre Z=0."""
    objp = np.zeros((FILAS * COLUMNAS, 3), np.float32)
    grid = np.mgrid[0:COLUMNAS, 0:FILAS].T.reshape(-1, 2)

    if centered:
        center = np.array([(COLUMNAS - 1) / 2.0, (FILAS - 1) / 2.0], dtype=np.float32)
        grid = grid - center

    objp[:, :2] = grid * TAM_CUADRADO
    return objp


def calibration_image_captures(output_dir="CalibrationImages", num_images=20, camera_index=0, interval_sec= 1.5):
    """Captura imagenes de calibracion y las guarda en output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # Contar imagenes existentes para no sobrescribir
    existing_images = []
    for ext in ("*.jpg", "*.JPG", "*.png", "*.jpeg"):
        existing_images.extend(glob.glob(os.path.join(output_dir, ext))) 
    next_index = len(existing_images) + 1 

    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        print("No se pudo abrir la camara para capturar imagenes de calibracion")
        return []

    print(f"Capturando {num_images} imagenes en la carpeta: {output_dir}")
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
        ret, corners = cv.findChessboardCorners(gray, (COLUMNAS, FILAS), None)

        preview = frame.copy()
        if ret:
            cv.drawChessboardCorners(preview, (COLUMNAS, FILAS), corners, ret)

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


def clear_calibration_images(output_dir="CalibrationImages"):
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
    carpeta_imagenes = "CalibrationImages"
    os.makedirs(carpeta_imagenes, exist_ok=True)

    extensions = ["*.jpg", "*.JPG", "*.png", "*.jpeg"]
    images = []
    for ext in extensions:
        ruta_busqueda = os.path.join(carpeta_imagenes, ext)
        images.extend(glob.glob(ruta_busqueda))

    if len(images) == 0:
        print("CalibrationImages esta vacia. Iniciando captura de 20 imagenes...")
        images = calibration_image_captures(output_dir=carpeta_imagenes, num_images = 20)
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
        ret, corners = cv.findChessboardCorners(gray, (COLUMNAS, FILAS), None)

        if ret:
            print(f"Patron encontrado en {fname}")
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERM_CRITERIA)
            imgpoints.append(corners2)

            cv.drawChessboardCorners(img, (COLUMNAS, FILAS), corners2, ret)
            # cv.imshow("Calibracion", img)
            # cv.waitKey(500)
        else:
            print(f"No se detecto el patron {COLUMNAS}x{FILAS} en {fname}")

    cv.destroyAllWindows()

    if len(objpoints) == 0 or len(imgpoints) == 0 or image_size is None:
        print("No se encontraron suficientes patrones validos para calibrar")
        return None, None, None, None, None

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    return ret, mtx, dist, rvecs, tvecs


def aruco_calibration():
    return None


def charuco_calibration():
    return None


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


def live_projection(mtx, dist, objp, puntos_modelo):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la camara")
        return

    print("Presiona 'q' para cerrar la proyeccion en vivo")

    # --- VARIABLES PARA OPTIMIZACIÓN ---
    frame_count = 0
    skip_rate = 6  # Si no hay patrón, solo busca cada 6 frames (~5 veces por segundo)
    patron_detectado = False 

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Solo buscamos si el patrón estaba presente o si toca por el contador de saltos
        debe_buscar = patron_detectado or (frame_count % skip_rate == 0)

        if debe_buscar:
            # CALIB_CB_FAST_CHECK hace una pasada rápida y aborta si no ve un tablero
            flags_opt = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
            ret_pattern, corners = cv.findChessboardCorners(gray, (COLUMNAS, FILAS), flags=flags_opt)

            if ret_pattern:
                patron_detectado = True # Mantenemos FPS altos
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERM_CRITERIA)
                ret_pnp, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

                if ret_pnp and puntos_modelo is not None:
                    imgpts, _ = cv.projectPoints(puntos_modelo, rvec, tvec, mtx, dist)
                    for pt in imgpts:
                        x, y = int(pt[0][0]), int(pt[0][1])
                        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

            else:
                patron_detectado = False # Si no se detecta, entramos en modo ahorro de CPU
                cv.putText(frame, "Buscando patron", (10, 30), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # En los frames saltados, mantenemos el feedback visual
            cv.putText(frame, "Buscando patron", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv.imshow("Realidad Aumentada PCD", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


def main(calibration_type="chessboard", show_intrinsic=False, show_extrinsic=False):
    print("=== PRACTICA 1: CALIBRACION Y PROYECCION ===")

    print(f"\nRealizando calibracion con tipo: {calibration_type.upper()}...")
    
    # Seleccionar funcion de calibracion segun el tipo
    if calibration_type == "chessboard":
        ret, mtx, dist, rvecs, tvecs = chessboard_calibration()
    elif calibration_type == "aruco":
        ret, mtx, dist, rvecs, tvecs = aruco_calibration()
    elif calibration_type == "charuco":
        ret, mtx, dist, rvecs, tvecs = charuco_calibration()
    else:
        print(f"Tipo de calibracion desconocido: {calibration_type}")
        return

    # Si el error de reproyeccion es alto, intentamos capturar nuevas imagenes y recalibrar
    reintentos = 0
    while mtx is not None and ret > RMS_UMBRAL_ALTO and reintentos < MAX_REINTENTOS_RMS:
        reintentos += 1
        print(
            f"\nAviso: error de reproyeccion alto ({ret:.6f} > {RMS_UMBRAL_ALTO:.3f})."
        )
        print(f"Reintento automatico {reintentos}/{MAX_REINTENTOS_RMS}")

        borradas = clear_calibration_images(output_dir="CalibrationImages")
        print(f"Se borraron {borradas} imagenes de CalibrationImages")

        nuevas = calibration_image_captures(
            output_dir="CalibrationImages",
            num_images=CAPTURAS_REINTENTO,
        )

        if len(nuevas) == 0:
            print("No se capturaron nuevas imagenes. No se puede repetir la calibracion")
            break

        print("Recalculando calibracion con las nuevas imagenes...")
        # Recalibrar usando el mismo tipo de calibracion
        if calibration_type == "chessboard":
            ret, mtx, dist, rvecs, tvecs = chessboard_calibration()
        elif calibration_type == "aruco":
            ret, mtx, dist, rvecs, tvecs = aruco_calibration()
        elif calibration_type == "charuco":
            ret, mtx, dist, rvecs, tvecs = charuco_calibration()

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
        objp = build_object_points(centered=True)
        escala_modelo = 0.05 * TAM_CUADRADO
        puntos_pcd = load_pcd_model("ninetales_voxelizado.pcd", escala=escala_modelo)
        live_projection(mtx, dist, objp, puntos_pcd)
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
