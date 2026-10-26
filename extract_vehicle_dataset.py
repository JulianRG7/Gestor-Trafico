from PIL import Image
import os

# ==================================================
# MAPEO DE CLASES (enum VehicleType de Unity)
# ==================================================

VEHICLE_TYPES = {
    0: "Car",
    1: "Motorcycle",
    2: "Passenger",
    3: "Cargo",
    4: "Ambulance",
    5: "Police",
    6: "FireTruck",
}

EXTENSIONES_IMAGEN = (".png", ".jpg", ".jpeg")


def _ruta_unica(ruta):
    """
    Si 'ruta' ya existe, le agrega un sufijo _1, _2, ...
    hasta encontrar un nombre libre.

    Con esto, aunque se repita accidentalmente un nombre
    de archivo, NUNCA se sobreescribe un recorte ya
    generado en una ejecucion anterior.
    """

    if not os.path.exists(ruta):
        return ruta

    base, ext = os.path.splitext(ruta)
    contador = 1

    while True:
        nueva_ruta = f"{base}_{contador}{ext}"
        if not os.path.exists(nueva_ruta):
            return nueva_ruta
        contador += 1


def _recortar_imagen(imagen_path, txt_path, carpeta_salida, tag_dataset):
    """
    Recorta todos los bounding boxes de UNA imagen y
    guarda cada recorte en carpeta_salida/<Clase>/...
    """

    if not os.path.exists(imagen_path):
        print(f"[ERROR] No existe la imagen: {imagen_path}")
        return 0

    if not os.path.exists(txt_path):
        print(f"[ADVERTENCIA] No existe el TXT para: {imagen_path}")
        return 0

    imagen = Image.open(imagen_path).convert("RGB")
    ancho, alto = imagen.size

    nombre_imagen = os.path.splitext(os.path.basename(imagen_path))[0]

    with open(txt_path, "r") as archivo:
        lineas = archivo.readlines()

    recortes_generados = 0

    # ==========================================
    # PROCESAR CADA VEHÍCULO DE LA IMAGEN
    # ==========================================

    for numero, linea in enumerate(lineas, start=1):

        linea = linea.strip()

        if not linea:
            continue

        datos = linea.split()

        if len(datos) < 5:
            print(f"[ADVERTENCIA] Linea {numero} invalida en {txt_path}:")
            print(linea)
            continue

        try:

            # Formato YOLO:
            # class_id x_center y_center width height

            class_id = int(datos[0])

            x_center = float(datos[1])
            y_center = float(datos[2])
            box_width = float(datos[3])
            box_height = float(datos[4])

            # ==========================================
            # COORDENADAS (igual que en el script original)
            # ==========================================

            # Unity: x=0 izquierda, x=1 derecha
            x_center_px = x_center * ancho

            # Unity: y=0 abajo, y=1 arriba
            # PIL:   y=0 arriba, y=alto abajo -> se invierte
            y_center_px = (1.0 - y_center) * alto

            box_width_px = box_width * ancho
            box_height_px = box_height * alto

            x1 = int(x_center_px - box_width_px / 2)
            y1 = int(y_center_px - box_height_px / 2)

            x2 = int(x_center_px + box_width_px / 2)
            y2 = int(y_center_px + box_height_px / 2)

            # Limitar al tamaño de la imagen
            x1 = max(0, min(ancho - 1, x1))
            y1 = max(0, min(alto - 1, y1))
            x2 = max(0, min(ancho - 1, x2))
            y2 = max(0, min(alto - 1, y2))

            if x2 <= x1 or y2 <= y1:
                print(
                    f"[ADVERTENCIA] Bounding box vacio en linea {numero} "
                    f"de {txt_path}, se omite."
                )
                continue

            # ==========================================
            # CARPETA DE LA CLASE
            # ==========================================

            nombre_clase = VEHICLE_TYPES.get(class_id, f"Clase_{class_id}")

            carpeta_clase = os.path.join(carpeta_salida, nombre_clase)
            os.makedirs(carpeta_clase, exist_ok=True)

            # ==========================================
            # RECORTAR Y GUARDAR
            # ==========================================

            recorte = imagen.crop((x1, y1, x2, y2))

            nombre_archivo = f"{tag_dataset}_{nombre_imagen}_{numero:03d}.jpg"
            ruta_salida = os.path.join(carpeta_clase, nombre_archivo)
            ruta_salida = _ruta_unica(ruta_salida)

            recorte.save(ruta_salida)
            recortes_generados += 1

            print(
                f"  Objeto {numero}: "
                f"Clase={class_id} ({nombre_clase}) | "
                f"Box=({x1}, {y1}) -> ({x2}, {y2}) | "
                f"{os.path.basename(ruta_salida)}"
            )

        except ValueError:
            print(
                f"[ERROR] No se pudo interpretar la linea {numero} "
                f"en {txt_path}: {linea}"
            )

    return recortes_generados


def recortar_dataset(carpeta_dataset, carpeta_salida):
    """
    Procesa TODAS las imagenes de un dataset:

        carpeta_dataset/images/*.png|jpg
        carpeta_dataset/labels/*.txt   (mismo nombre base)

    Los recortes quedan en:

        carpeta_salida/<NombreClase>/<tag_dataset>_<imagen>_<idx>.jpg

    'tag_dataset' se toma automaticamente del nombre de la
    carpeta del dataset (ej. "YOLODataset-c2-a1"), asi que
    puedes apuntar este mismo 'carpeta_salida' desde varios
    datasets/angulos distintos sin que se pisen los archivos,
    aunque las imagenes se llamen igual (image_000005.png, etc).
    """

    carpeta_imagenes = os.path.join(carpeta_dataset, "images")
    carpeta_labels = os.path.join(carpeta_dataset, "labels")

    if not os.path.isdir(carpeta_imagenes):
        print(f"ERROR: No existe la carpeta de imagenes:")
        print(carpeta_imagenes)
        return

    if not os.path.isdir(carpeta_labels):
        print(f"ERROR: No existe la carpeta de labels:")
        print(carpeta_labels)
        return

    tag_dataset = os.path.basename(os.path.normpath(carpeta_dataset))

    archivos_imagen = sorted(
        f for f in os.listdir(carpeta_imagenes)
        if f.lower().endswith(EXTENSIONES_IMAGEN)
    )

    print("==========================================")
    print(f"DATASET: {tag_dataset}")
    print(f"Imagenes encontradas: {len(archivos_imagen)}")
    print("==========================================")
    print()

    total_recortes = 0

    for archivo_imagen in archivos_imagen:

        nombre_base = os.path.splitext(archivo_imagen)[0]

        imagen_path = os.path.join(carpeta_imagenes, archivo_imagen)
        txt_path = os.path.join(carpeta_labels, nombre_base + ".txt")

        print(f"[{nombre_base}]")

        total_recortes += _recortar_imagen(
            imagen_path,
            txt_path,
            carpeta_salida,
            tag_dataset,
        )

        print()

    print("==========================================")
    print("PROCESO TERMINADO")
    print(f"Total de recortes generados: {total_recortes}")
    print(f"Carpeta de salida: {carpeta_salida}")
    print("==========================================")


# ==================================================
# RUTAS
# ==================================================
# Cambia 'carpeta_dataset' manualmente en cada ejecucion
# (YOLODataset, YOLODataset-c2-a1, YOLODataset-c3-a1,
# YOLODataset-cf-a2, etc).
#
# 'carpeta_salida' NO la cambies: se mantiene igual siempre
# para que los recortes de todos los angulos se acumulen
# juntos, organizados por clase, sin sobreescribirse.
# ==================================================

carpeta_dataset = r"C:\Users\Esteban_Julian\Documents\UG\Semeste_8vo\SUMO2Unity-main\YOLODataset-Camera_Free3-L-V2"
carpeta_salida = r"C:\Users\Esteban_Julian\Documents\UG\Semeste_8vo\SUMO2Unity-main\VehicleClassificationDataset"

recortar_dataset(carpeta_dataset, carpeta_salida)