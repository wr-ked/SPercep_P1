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

    # Criterio de terminación
    termcriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 
    
    # Preparar puntos de objeto.
    objp = np.zeros((filas*columnas, 3), np.float32)
    objp[:,:2] = np.mgrid[0:filas, 0:columnas].T.reshape(-1,2)

    objpoints = [] # Puntos 3D
    imgpoints = [] # Puntos 2D

    # Buscar imágenes en la carpeta "Chess_Calibration_Images". Si etsa vacia abre la camara para hacer la calibracion en vivo.
    carpeta_imagenes = "Chess_Calibration_Images"
    extensions = ['*.jpg', '*.JPG', '*.png', '*.jpeg']
    images = []
    for ext in extensions:
        ruta_busqueda = os.path.join(carpeta_imagenes, ext)
        images.extend(glob.glob(ruta_busqueda))

    if len(images) == 0:
        print("No se encontraron imágenes de calibración. Abriendo cámara para calibración en vivo...")
        calibration_image_captures()
        return None, None
    
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print(f"Error leyendo {fname}")
            continue
    
        # Escala de grises para la detección de esquinas
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

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

    # CALIBRACION DE LA CAMARA
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

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
    mtx, dist = chessboard_calibration()

    if mtx is not None:
        print("\n¡Calibración exitosa!")
        print("Iniciando captura de video en vivo...")
        # Aquí irá la función de video en vivo
        # captura_video_en_vivo(mtx, dist)
    else:
        print("No se puede continuar sin calibración válida")

if __name__ == "__main__":
    main()