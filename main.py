# Perception. Practice 1. Camera calibration and 3D to 2D projection
# Authors: Hugo Villasan Atienza y Diego Lopez Salazar

import numpy as np
import cv2 as cv
import glob
import os

try:
    import open3d as o3d
except ImportError:
    o3d = None


# Parametros globales
FILAS = 9
COLUMNAS = 6
TAM_CUADRADO = 0.02  # metros
TERM_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def build_object_points(centered=True):
    """Genera los puntos 3D del tablero sobre Z=0."""
    objp = np.zeros((FILAS * COLUMNAS, 3), np.float32)
    grid = np.mgrid[0:COLUMNAS, 0:FILAS].T.reshape(-1, 2)

    if centered:
        center = np.array([(COLUMNAS - 1) / 2.0, (FILAS - 1) / 2.0], dtype=np.float32)
        grid = grid - center

    objp[:, :2] = grid * TAM_CUADRADO
    return objp


def calibration_image_captures():
    # TODO: Implementar esta funcion para capturar imagenes en vivo.
    return None


def chessboard_calibration():
    # Puntos 3D del patron (centrados para facilitar la proyeccion AR)
    objp = build_object_points(centered=True)

    objpoints = []
    imgpoints = []
    image_size = None

    # Buscar imagenes de calibracion en carpetas candidatas
    carpetas_candidatas = [ "CalibracionPCDiego"]
    extensions = ["*.jpg", "*.JPG", "*.png", "*.jpeg"]
    images = []
    carpeta_imagenes = None

    for carpeta in carpetas_candidatas:
        if not os.path.exists(carpeta):
            continue

        images_temp = []
        for ext in extensions:
            ruta_busqueda = os.path.join(carpeta, ext)
            images_temp.extend(glob.glob(ruta_busqueda))

        if images_temp:
            carpeta_imagenes = carpeta
            images = images_temp
            break

    if carpeta_imagenes is None:
        print("No existe ninguna carpeta de calibracion valida")
        return None, None, None, None, None

    print(f"Se usaran {len(images)} imagenes de: {carpeta_imagenes}")

    if len(images) == 0:
        print("No se encontraron imagenes de calibracion. Abriendo camara para calibracion en vivo...")
        calibration_image_captures()
        return None, None, None, None, None

    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print(f"Error leyendo {fname}")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            print(f"Se omite {fname}: tamano de imagen inconsistente")
            continue

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

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret_pattern, corners = cv.findChessboardCorners(gray, (COLUMNAS, FILAS), None)

        if ret_pattern:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERM_CRITERIA)
            ret_pnp, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

            if ret_pnp and puntos_modelo is not None:
                imgpts, _ = cv.projectPoints(puntos_modelo, rvec, tvec, mtx, dist)
                for pt in imgpts:
                    x, y = int(pt[0][0]), int(pt[0][1])
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

            cv.drawChessboardCorners(frame, (COLUMNAS, FILAS), corners2, ret_pattern)
        else:
            cv.putText(frame, "Buscando patron...", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv.imshow("Realidad Aumentada PCD", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


def main():
    print("=== PRACTICA 1: CALIBRACION Y PROYECCION ===")

    print("\nRealizando calibracion con patron de ajedrez...")
    ret, mtx, dist, rvecs, tvecs = chessboard_calibration()

    if mtx is not None:
        print("\nCalibracion exitosa")
        print("\n=== MATRIZ DE PARAMETROS INTRINSECOS (M) ===")
        print("Matriz de camara:")
        print(mtx)
        print(f"\nDistancia focal fx: {mtx[0,0]:.2f}")
        print(f"Distancia focal fy: {mtx[1,1]:.2f}")
        print(f"Centro optico cx: {mtx[0,2]:.2f}")
        print(f"Centro optico cy: {mtx[1,2]:.2f}")

        print("\nCoeficientes de distorsion:")
        print(dist.ravel())

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
    main()
