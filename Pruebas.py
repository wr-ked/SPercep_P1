import numpy as np
import cv2 as cv
import glob
import os
import open3d as o3d  # <-- NUEVO: Necesario para leer el archivo .pcd binario

# --- PARÁMETROS DE LA PRÁCTICA ---
FILAS = 9
COLUMNAS = 6

# Criterio de terminación
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparar puntos de objeto centrados en (0,0,0)
tam_cuadrado = 0.02
objp = np.zeros((FILAS*COLUMNAS, 3), np.float32)

# 1. Crear la cuadrícula original
grid = np.mgrid[0:FILAS, 0:COLUMNAS].T.reshape(-1, 2)

# 2. Calcular dónde está el centro matemático en "cuadros"
centro_x = (FILAS - 1) / 2.0  # Para 9 filas, el centro es 4.0
centro_y = (COLUMNAS - 1) / 2.0  # Para 6 columnas, el centro es 2.5

# 3. Restar el centro a la cuadrícula para que el origen quede en medio
grid_centrado = grid - np.array([centro_x, centro_y])

# 4. Asignar al array 3D y escalar por el tamaño real del cuadro
objp[:, :2] = grid_centrado * tam_cuadrado

objpoints = [] # Puntos 3D
imgpoints = [] # Puntos 2D

# --- CORRECCIÓN DE EXTENSIONES ---
carpeta_imagenes = "CalibracionPCDiego"
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
    ret, corners = cv.findChessboardCorners(gray, (FILAS, COLUMNAS), None)

    if ret == True:
        print(f"OK: Patrón encontrado en {fname}")
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # cv.drawChessboardCorners(img, (FILAS, COLUMNAS), corners2, ret)
        # cv.imshow('Calibracion', img)
        # cv.waitKey(500) 
    else:
        print(f"FALLO: No se detectó el patrón 9x6 en {fname}")

cv.destroyAllWindows()

# --- CALIBRACIÓN DE LA CÁMARA ---
if len(objpoints) > 0 and len(imgpoints) > 0:
    print(f"\n--- CALIBRACIÓN CON {len(objpoints)} IMÁGENES ---")
    img = cv.imread(images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    if ret:
        print("\n=== MATRIZ DE PARÁMETROS INTRÍNSECOS (M) ===")
        print(mtx)
        print(f"\nError de reproyección RMS: {ret:.6f}")
    else:
        print("ERROR: Fallo en la calibración de la cámara")
        exit()
else:
    print("ERROR: No se encontraron suficientes patrones válidos para calibrar")
    exit()

# --- CARGAR MODELO 3D (.PCD) ---
def cargar_modelo(ruta_pcd, escala=1.0):

    try:
        pcd = o3d.io.read_point_cloud(ruta_pcd)
        puntos_3d = np.asarray(pcd.points)
        
        # Escalar los puntos y asegurarnos de que el tipo sea float32 para OpenCV
        puntos_3d = np.float32(puntos_3d * escala)
        print(f"--> Modelo cargado correctamente con {len(puntos_3d)} puntos.")
        return puntos_3d
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo PCD. Detalles: {e}")
        return None


# Definimos el factor de escala (hay que ajustarlo probando)
ESCALA_MODELO = 0.05
puntos_pcd = cargar_modelo("ninetales_centrado.pcd", escala=ESCALA_MODELO*tam_cuadrado)

# puntos_modelo = crear_linea_horizontal(num_puntos=50)

# --- VIDEO EN VIVO Y PROYECCIÓN AR ---
def captura_video_en_vivo(mtx, dist, objp, puntos_modelo):
    """
    Captura video, calcula la pose de la cámara y proyecta el modelo 3D.
    """
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: No se puede abrir la cámara")
        return
    
    print("Presiona 'q' para salir del video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret_pattern, corners = cv.findChessboardCorners(gray, (FILAS, COLUMNAS), None)
        
        if ret_pattern:
            # 1. Refinar las esquinas
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            # 2. Encontrar la matriz de transformación de la pose (solvePnP)
            ret_pnp, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)
            
            # 3. Proyectar y dibujar la nube de puntos si tenemos modelo y pose válida
            if ret_pnp and puntos_modelo is not None:
                # Proyectar los puntos 3D del modelo al plano 2D de la imagen
                imgpts, _ = cv.projectPoints(puntos_modelo, rvec, tvec, mtx, dist)
                
                # Dibujar los puntos proyectados en el frame
                for pt in imgpts:
                    x, y = int(pt[0][0]), int(pt[0][1])
                    
                    # Comprobar que el punto cae dentro de la pantalla para evitar errores
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        cv.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Puntos rojos
            
            # Opcional: Descomenta esto si quieres ver también los puntos del patrón
            cv.drawChessboardCorners(frame, (FILAS, COLUMNAS), corners2, ret_pattern)
            
        else:
            cv.putText(frame, 'Buscando patron...', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv.imshow('Realidad Aumentada PCD', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()

# Ejecutar el video
captura_video_en_vivo(mtx, dist, objp, puntos_pcd)
# captura_video_en_vivo(mtx, dist, objp, puntos_modelo)