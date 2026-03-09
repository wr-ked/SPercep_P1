# Perception. Practice 1. Camera calibration and 3D to 2D projection
# Authors: Hugo Villasan Atienza and Diego Lopez Salazar

import numpy as np
import cv2 as cv
import glob
import os


def chessboard_calibration(): 
   return None

def aruco_calibration():
    return None

def charuco_calibration():
    return None

def live_proyection():
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