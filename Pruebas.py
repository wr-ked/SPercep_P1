import numpy as np
import cv2 as cv
import glob
import os

# --- PARÁMETROS DE LA PRÁCTICA ---
# El PDF indica 9 intersecciones por fila y 6 por columna 
FILAS = 9
COLUMNAS = 6

# Criterio de terminación
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparar puntos de objeto (0,0,0), (1,0,0)...
objp = np.zeros((FILAS*COLUMNAS, 3), np.float32)
objp[:,:2] = np.mgrid[0:FILAS, 0:COLUMNAS].T.reshape(-1,2)

objpoints = [] # Puntos 3D
imgpoints = [] # Puntos 2D

# --- CORRECCIÓN DE EXTENSIONES ---
# Buscamos .jpg, .JPG, .png y .jpeg para asegurar que las pilla todas
extensions = ['*.jpg', '*.JPG', '*.png', '*.jpeg']
images = []
for ext in extensions:
    images.extend(glob.glob(ext))

print(f"--> Se han encontrado {len(images)} imágenes en la carpeta.")

if len(images) == 0:
    print("ERROR: No se encontraron imágenes. Revisa que el script esté en la misma carpeta que las fotos.")
    exit()

for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Error leyendo {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Buscar esquinas con las dimensiones correctas (9x6)
    ret, corners = cv.findChessboardCorners(gray, (FILAS, COLUMNAS), None)

    if ret == True:
        print(f"OK: Patrón encontrado en {fname}")
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Dibujar esquinas
        cv.drawChessboardCorners(img, (FILAS, COLUMNAS), corners2, ret)
        cv.imshow('Calibracion', img)
        
        # --- CORRECCIÓN DEL BLOQUEO ---
        # cv.waitKey(500) espera 500ms (medio segundo) y sigue solo.
        # Si pones 0, se congela hasta que pulses una tecla.
        cv.waitKey(500) 
    else:
        print(f"FALLO: No se detectó el patrón 9x6 en {fname}")

cv.destroyAllWindows()
print("Proceso finalizado.")