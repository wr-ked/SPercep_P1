# SPercep_P1

## Introduccion

Esta practica aborda la calibracion de una camara para obtener sus parametros intrinsecos y extrinsecos, y usar esos resultados en una proyeccion 3D sobre imagen 2D. La calibracion es el paso base para tareas de vision por computador como reconstruccion, medicion y realidad aumentada.

## Estado del Arte

El enfoque mas extendido en entornos academicos e industriales sigue siendo la calibracion geometrica con patrones conocidos y minimizacion del error de reproyeccion. El metodo de Zhang y sus variantes practicas se han consolidado por su robustez y bajo coste: solo requieren multiples vistas del patron en distintas poses.

Actualmente, OpenCV ofrece implementaciones maduras para tres familias principales:

- Chessboard: muy estable y simple para escenarios controlados.

- ArUco: util cuando se necesita deteccion robusta de marcadores individuales.

- ChArUco: combina esquinas tipo ajedrez con IDs ArUco, logrando buena precision y trazabilidad.

## Desarrollo

### Calibracion de la camara

El flujo seguido es:

1. Captura de imagenes del patron en diferentes orientaciones y posiciones.

2. Deteccion de puntos caracteristicos (esquinas o marcadores).

3. Estimacion de la matriz intrinseca, coeficientes de distorsion y poses extrinsecas.

4. Evaluacion de calidad mediante error de reproyeccion RMS.

### Modos disponibles

El script permite elegir de forma obligatoria el tipo de calibracion con el flag `-c`:

- Chessboard
- ArUco
- ChArUco

Ejemplos:

```bash
python main.py -c chessboard
python main.py -c aruco
python main.py -c charuco
python main.py -c charuco -I -E
```

Donde:

- `-I` muestra parametros intrinsecos.
- `-E` muestra parametros extrinsecos.

### Carpetas de calibracion

Para evitar mezclar datasets, cada patron usa su propia carpeta de imagenes:

- Chessboard: `CalibrationImagesChessboard`
- ArUco: `CalibrationImagesAruco`
- ChArUco: `CalibrationImagesCharuco`

Criterio de calidad usado:

- RMS < 0.8: calibracion aceptable (error medio menor que 1 pixel).

- RMS < 0.5: calibracion optima.

Ademas, si el RMS supera el umbral configurado, el sistema puede repetir automaticamente la captura y recalibrar para mejorar la calidad.

## Experimentacion

Se realizaron capturas desde multiples angulos y distancias para mejorar la condicion numerica del problema. Tras la calibracion, se evaluo el RMS global y se compararon resultados entre intentos para validar estabilidad.

## Conclusiones

La calibracion permite pasar de una camara "desconocida" a un modelo metrico util para proyeccion y estimacion de pose. Mantener diversidad en las capturas reduce el error de reproyeccion y mejora la robustez del sistema.

## Bibliografia

- OpenCV Documentation. Camera Calibration. <https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html>
