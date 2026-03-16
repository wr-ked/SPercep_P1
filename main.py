# Perception. Practice 1. Camera calibration and 3D to 2D projection
# Authors: Hugo Villasan Atienza and Diego Lopez Salazar

import numpy as np
import cv2 as cv
import glob
import os


def calibration_image_captures():       # TODO: Implementar esta función para capturar imágenes de calibración en vivo si no se encuentran imágenes en la carpeta.
    return None

def chessboard_calibration():

    # Dimensiones del patrón de ajedrez
    filas = 9
    columnas = 6
    tam_cuadrado = 0.02 # centímetros, tamaño real de cada cuadrado del patrón

    # Criterio de terminación
    termcriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 
    
    # Preparar puntos de objeto.
    objp = np.zeros((filas*columnas, 3), np.float32)
    objp[:,:2] = np.mgrid[0:filas, 0:columnas].T.reshape(-1,2) * tam_cuadrado  # Multiplicamos por el tamaño del cuadrado para obtener coordenadas reales en centímetros

    objpoints = [] # Puntos 3D
    imgpoints = [] # Puntos 2D
    image_size = None

    # Buscar imágenes en la carpeta "Chess_Calibration_Images". Si esta vacia abre la camara para hacer la calibracion en vivo.
    carpeta_imagenes = "Chess_Calibration_Images"
    extensions = ['*.jpg', '*.JPG', '*.png', '*.jpeg']
    images = []
    for ext in extensions:
        ruta_busqueda = os.path.join(carpeta_imagenes, ext)
        images.extend(glob.glob(ruta_busqueda))

    if len(images) == 0:
        print("No se encontraron imágenes de calibración. Abriendo cámara para calibración en vivo...")
        calibration_image_captures()
        return None, None, None, None, None
    
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print(f"Error leyendo {fname}")
            continue
    
        # Escala de grises para la detección de esquinas
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        # Buscar esquinas con las dimensiones correctas (9x6)
        ret, corners = cv.findChessboardCorners(gray, (filas, columnas), None)
        
        if ret == True:
            # Si encuentra el patrón, lo añadimos a la lista de puntos de objeto
            print(f"Patrón encontrado en {fname}")
            objpoints.append(objp)

            # Refinamos las esquinas para mayor precisión y las añadimos a la lista de puntos de imagen
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), termcriteria)
            imgpoints.append(corners2)

            # Dibujar esquinas detectadas en la imagen
            cv.drawChessboardCorners(img, (filas, columnas), corners2, ret)
            cv.imshow('Calibración', img)
            cv.waitKey(500)  # Esperar medio segundo para mostrar la imagen
        else:
            print(f"No se detectó el patrón 9x6 en {fname}")

    cv.destroyAllWindows()

    if len(objpoints) == 0 or len(imgpoints) == 0 or image_size is None:
        print("No se encontraron suficientes patrones válidos para calibrar")
        return None, None, None, None, None

    # CALIBRACION DE LA CAMARA
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    return ret, mtx, dist, rvecs, tvecs

    # VERBOSE
    

def aruco_calibration():
    return None

def charuco_calibration():
    return None

def live_projection():
    return None

def main():
    print("=== PRÁCTICA 1: CALIBRACIÓN Y REALIDAD AUMENTADA ===")
    
    # Ejecutar calibración chessboard
    print("\nRealizando calibración con patrón de ajedrez...")
    ret, mtx, dist, rvecs, tvecs = chessboard_calibration()

    if mtx is not None:
        print("\n¡Calibración exitosa!")
        print("\n=== MATRIZ DE PARÁMETROS INTRÍNSECOS (M) ===")
        print("Matriz de cámara:")
        print(mtx)
        print(f"\nDistancia focal fx: {mtx[0,0]:.2f}")
        print(f"Distancia focal fy: {mtx[1,1]:.2f}")
        print(f"Centro óptico cx: {mtx[0,2]:.2f}")
        print(f"Centro óptico cy: {mtx[1,2]:.2f}")

        print("\nCoeficientes de distorsión:")
        print(dist.ravel())

        print("\n=== PARÁMETROS EXTRÍNSECOS PARA CADA IMAGEN ===")
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            print(f"\n--- IMAGEN {i+1} ---")

            # Convertir vector de rotación a matriz de rotación
            R, _ = cv.Rodrigues(rvec)

            print("Matriz de rotación R:")
            print(R)

            print("Vector de traslación T:")
            print(tvec.ravel())

            # Matriz de transformación completa 4x4
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.ravel()

            print("Matriz de transformación completa T (4x4):")
            print(T)

        print(f"\n=== RESUMEN DE CALIBRACIÓN ===")
        print(f"Error de reproyección RMS: {ret:.6f}")
        print(f"Imágenes utilizadas: {len(rvecs)}")

        print("\nIniciando captura de video en vivo...")
        # Aquí irá la función de video en vivo
        # captura_video_en_vivo(mtx, dist)
    else:
        print("No se puede continuar sin calibración válida")

if __name__ == "__main__":
    main()