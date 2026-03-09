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
# Ahora buscamos en la subcarpeta "Imagenes calibrador"
carpeta_imagenes = "Calibration_Images"
extensions = ['*.jpg', '*.JPG', '*.png', '*.jpeg']
images = []
for ext in extensions:
    ruta_busqueda = os.path.join(carpeta_imagenes, ext)
    images.extend(glob.glob(ruta_busqueda))

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

# --- CALIBRACIÓN DE LA CÁMARA ---
if len(objpoints) > 0 and len(imgpoints) > 0:
    print(f"\n--- CALIBRACIÓN CON {len(objpoints)} IMÁGENES ---")
    
    # Obtener dimensiones de imagen (usar la última imagen válida)
    img = cv.imread(images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Calibrar la cámara
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    if ret:
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
        print(f"Imágenes utilizadas: {len(objpoints)}")
        
    else:
        print("ERROR: Fallo en la calibración de la cámara")
else:
    print("ERROR: No se encontraron suficientes patrones válidos para calibrar")

print("Proceso finalizado.")

def captura_video_en_vivo(mtx, dist):
    """
    Captura video en vivo y detecta el patrón de calibración
    mtx: matriz de parámetros intrínsecos
    dist: coeficientes de distorsión
    """
    # Inicializar la cámara (0 es la cámara por defecto)
    cap = cv.VideoCapture(0)
    
    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("ERROR: No se puede abrir la cámara")
        return
    
    print("Presiona 'q' para salir del video")
    print("Presiona 'c' para capturar una imagen")
    
    while True:
        # Capturar frame
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: No se puede recibir frame de la cámara")
            break
        
        # Convertir a escala de grises
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Buscar el patrón de calibración
        ret_pattern, corners = cv.findChessboardCorners(gray, (FILAS, COLUMNAS), None)
        
        # Si se encuentra el patrón
        if ret_pattern:
            # Refinar las esquinas
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            # Dibujar las esquinas
            cv.drawChessboardCorners(frame, (FILAS, COLUMNAS), corners2, ret_pattern)
            
            # Mostrar texto indicando que se detectó el patrón
            cv.putText(frame, 'Patron detectado', (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Mostrar texto indicando que no se detectó
            cv.putText(frame, 'Buscando patron...', (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Mostrar el frame
        cv.imshow('Video en vivo - Deteccion de patron', frame)
        
        # Verificar teclas presionadas
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capturar imagen actual
            cv.imwrite('captura_actual.jpg', frame)
            print("Imagen capturada como 'captura_actual.jpg'")
    
    # Liberar recursos
    cap.release()
    cv.destroyAllWindows()

captura_video_en_vivo(mtx, dist)