import open3d as o3d
import numpy as np
import os

def simplificar_pcd(ruta_entrada, ruta_salida, tamano_voxel):
    # 1. Comprobar que el archivo existe
    if not os.path.exists(ruta_entrada):
        print(f"ERROR: No se encuentra el archivo '{ruta_entrada}'")
        return

    print(f"Cargando modelo original: {ruta_entrada}...")
    
    # 2. Leer la nube de puntos original
    pcd = o3d.io.read_point_cloud(ruta_entrada)
    puntos_originales = len(pcd.points)
    print(f"--> Puntos antes del filtro: {puntos_originales}")

    if puntos_originales == 0:
        print("ERROR: El modelo está vacío o no se pudo leer correctamente.")
        return

    # 3. Aplicar el filtro Voxel Grid
    print(f"Aplicando Voxel Grid con tamaño: {tamano_voxel}...")
    pcd_filtrado = pcd.voxel_down_sample(voxel_size=tamano_voxel)
    puntos_nuevos = len(pcd_filtrado.points)
    
    print(f"--> Puntos después del filtro: {puntos_nuevos}")
    
    # Calcular porcentaje de reducción
    reduccion = (1 - (puntos_nuevos / puntos_originales)) * 100
    print(f"--> Se ha reducido la cantidad de puntos en un {reduccion:.2f}%")

    # 4. Guardar el nuevo modelo optimizado
    o3d.io.write_point_cloud(ruta_salida, pcd_filtrado)
    print(f"\n¡Éxito! Nuevo modelo guardado en: {ruta_salida}")

    # 5. Opcional: Visualizar el resultado
    # Descomenta la siguiente línea si quieres que se abra una ventana 3D para ver cómo quedó
    # o3d.visualization.draw_geometries([pcd_filtrado], window_name="Modelo Voxelizado")

# --- CONFIGURACIÓN ---
# Nombre de tu archivo original
archivo_original = "ninetales_centrado.pcd"

# Nombre del nuevo archivo más ligero que vas a generar
archivo_optimizado = "ninetales_voxelizado.pcd"

# Ajusta este valor. 
# Si el modelo original está en milímetros o es muy grande, quizás necesites 1.0, 5.0, etc.
# Si el modelo está normalizado cerca de 1.0, valores como 0.01 o 0.05 son mejores.
TAMAÑO_VOXEL = 4

# Ejecutar la función
simplificar_pcd(archivo_original, archivo_optimizado, TAMAÑO_VOXEL)