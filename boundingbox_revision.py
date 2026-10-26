from PIL import Image, ImageDraw, ImageFont
import os


def dibujar_bounding_boxes(imagen_path, txt_path, salida_path=None):

    # ==========================================
    # VERIFICAR ARCHIVOS
    # ==========================================

    if not os.path.exists(imagen_path):
        print(f"ERROR: No existe la imagen:")
        print(imagen_path)
        return

    if not os.path.exists(txt_path):
        print(f"ERROR: No existe el archivo TXT:")
        print(txt_path)
        return

    # ==========================================
    # CARGAR IMAGEN
    # ==========================================

    imagen = Image.open(imagen_path).convert("RGB")
    ancho, alto = imagen.size

    draw = ImageDraw.Draw(imagen)

    print(f"Imagen: {ancho} x {alto}")

    # ==========================================
    # LEER TXT
    # ==========================================

    with open(txt_path, "r") as archivo:
        lineas = archivo.readlines()

    print(f"Objetos encontrados: {len(lineas)}")

    # ==========================================
    # PROCESAR CADA VEHÍCULO
    # ==========================================

    for numero, linea in enumerate(lineas, start=1):

        linea = linea.strip()

        if not linea:
            continue

        datos = linea.split()

        if len(datos) < 5:
            print(f"[ADVERTENCIA] Línea {numero} inválida:")
            print(linea)
            continue

        try:

            # Formato YOLO:
            #
            # class_id
            # x_center
            # y_center
            # width
            # height

            class_id = int(datos[0])

            x_center = float(datos[1])
            y_center = float(datos[2])
            box_width = float(datos[3])
            box_height = float(datos[4])

            # ==========================================
            # COORDENADAS X
            # ==========================================

            # Unity y YOLO:
            # x=0 izquierda
            # x=1 derecha

            x_center_px = x_center * ancho

            # ==========================================
            # COORDENADAS Y
            # ==========================================

            # IMPORTANTE:
            #
            # Unity:
            # y=0 -> abajo
            # y=1 -> arriba
            #
            # PIL:
            # y=0 -> arriba
            # y=alto -> abajo
            #
            # Por eso invertimos Y.

            y_center_px = (1.0 - y_center) * alto

            # ==========================================
            # TAMAÑO DEL BOUNDING BOX
            # ==========================================

            box_width_px = box_width * ancho
            box_height_px = box_height * alto

            # ==========================================
            # CALCULAR ESQUINAS
            # ==========================================

            x1 = int(x_center_px - box_width_px / 2)
            y1 = int(y_center_px - box_height_px / 2)

            x2 = int(x_center_px + box_width_px / 2)
            y2 = int(y_center_px + box_height_px / 2)

            # ==========================================
            # LIMITAR AL TAMAÑO DE LA IMAGEN
            # ==========================================

            x1 = max(0, min(ancho - 1, x1))
            y1 = max(0, min(alto - 1, y1))

            x2 = max(0, min(ancho - 1, x2))
            y2 = max(0, min(alto - 1, y2))

            # ==========================================
            # DIBUJAR BOUNDING BOX
            # ==========================================

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

            # ==========================================
            # DIBUJAR CLASE
            # ==========================================

            texto = f"Clase {class_id}"

            draw.text(
                (x1, max(0, y1 - 18)),
                texto,
                fill="red"
            )

            # ==========================================
            # INFORMACIÓN EN CONSOLA
            # ==========================================

            print(
                f"Objeto {numero}: "
                f"Clase={class_id} | "
                f"Centro normalizado=({x_center:.4f}, {y_center:.4f}) | "
                f"Centro píxeles=({x_center_px:.1f}, {y_center_px:.1f}) | "
                f"Box=({x1}, {y1}) -> ({x2}, {y2})"
            )

        except ValueError:

            print(
                f"[ERROR] No se pudo interpretar "
                f"la línea {numero}: {linea}"
            )

    # ==========================================
    # RUTA DE SALIDA
    # ==========================================

    if salida_path is None:

        nombre = os.path.splitext(
            os.path.basename(imagen_path)
        )[0]

        carpeta = os.path.dirname(imagen_path)

        salida_path = os.path.join(
            
            nombre + "_bbox.jpg"
        )

    # ==========================================
    # GUARDAR
    # ==========================================

    imagen.save(salida_path)

    print()
    print("==========================================")
    print("IMAGEN GENERADA CORRECTAMENTE")
    print("==========================================")
    print(salida_path)


# ==================================================
# RUTAS
# ==================================================

imagen_path = r"C:\Users\Esteban_Julian\Documents\UG\Semeste_8vo\SUMO2Unity-main\YOLODataset\images\image_000230.png"
txt_path = r"C:\Users\Esteban_Julian\Documents\UG\Semeste_8vo\SUMO2Unity-main\YOLODataset\labels\image_000230.txt"

dibujar_bounding_boxes(
    imagen_path,
    txt_path
)