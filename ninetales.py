import open3d as o3d
import numpy as np

# 1. Cargar la nube de puntos
nombre_archivo = "ninetales_invertido.pcd"
pcd = o3d.io.read_point_cloud(nombre_archivo)

# 2. Obtener las coordenadas
points = np.asarray(pcd.points)

# 3. Calcular los límites (bounding box)
min_bound = points.min(axis=0)
max_bound = points.max(axis=0)
# 4. Calcular el centro en X e Y
center_x = (min_bound[0] + max_bound[0]) / 2.0
center_y = (min_bound[1] + max_bound[1]) / 2.0

# Como el modelo está invertido y la cabeza era el mínimo, 
# la base (los pies) debe ser el valor MÁXIMO del eje Z.
base_z = max_bound[2] 

# 5. Calcular el vector de traslación para mover LA BASE al origen
translation = np.array([-center_x, -center_y, -base_z])

# 6. Aplicar la traslación a la nube de puntos
pcd.translate(translation)

# 7. Guardar el nuevo archivo
nuevo_archivo = "ninetales_centrado.pcd"
o3d.io.write_point_cloud(nuevo_archivo, pcd)

print(f"¡Listo! Archivo guardado como {nuevo_archivo}")