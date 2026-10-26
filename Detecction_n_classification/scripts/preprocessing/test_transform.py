from pathlib import Path
import random
import sys
import os
from PIL import Image, ImageDraw

# Sube dos niveles en las carpetas para apuntar a la raíz del proyecto
ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(ruta_raiz)

from datasets.transforms import ResizeAndPad


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "VehicleClassificationDataset"

OUTPUT_PATH = PROJECT_ROOT / "data" / "resize_pad_test"

IMAGES_PER_CLASS = 10

TARGET_SIZE = 240

# True:
#   Las imágenes pequeñas también pueden aumentar de tamaño.
#
# False:
#   Las imágenes menores de 240 permanecen en su tamaño original
#   y únicamente reciben padding.
ALLOW_UPSCALE = True

# Semilla para que la selección aleatoria sea reproducible.
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# FUNCIONES
# ============================================================

def get_image_files(class_dir):
    """
    Obtiene todas las imágenes de una clase.
    """

    return [
        path
        for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def create_comparison(original, transformed):
    """
    Crea una imagen de comparación:

        ORIGINAL | TRANSFORMADA

    Ambas se muestran sobre un canvas.
    """

    comparison_width = original.width + transformed.width
    comparison_height = max(
        original.height,
        transformed.height
    ) + 30

    canvas = Image.new(
        "RGB",
        (comparison_width, comparison_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    # Texto
    draw.text(
        (5, 5),
        "Original",
        fill="black"
    )

    draw.text(
        (original.width + 5, 5),
        "Resize + Pad",
        fill="black"
    )

    # Imágenes
    canvas.paste(
        original,
        (0, 30)
    )

    canvas.paste(
        transformed,
        (original.width, 30)
    )

    return canvas


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PRUEBA: RESIZE + PADDING")
    print("=" * 70)

    # --------------------------------------------------------
    # Verificar dataset
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        print(
            f"\n[ERROR] No se encontró el dataset:\n"
            f"{DATASET_PATH}"
        )

        return

    # --------------------------------------------------------
    # Crear directorio de salida
    # --------------------------------------------------------

    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Semilla
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Transformación
    # --------------------------------------------------------

    transform = ResizeAndPad(
        target_size=TARGET_SIZE,
        allow_upscale=ALLOW_UPSCALE
    )

    # --------------------------------------------------------
    # Obtener clases
    # --------------------------------------------------------

    class_dirs = sorted([
        directory
        for directory in DATASET_PATH.iterdir()
        if directory.is_dir()
        and directory.name != "analysis"
    ])

    print(f"\nDataset:")
    print(DATASET_PATH)

    print(f"\nClases encontradas: {len(class_dirs)}")

    # --------------------------------------------------------
    # Procesar cada clase
    # --------------------------------------------------------

    total_processed = 0

    for class_dir in class_dirs:

        class_name = class_dir.name

        image_files = get_image_files(class_dir)

        if len(image_files) < IMAGES_PER_CLASS:

            print(
                f"\n[WARNING] {class_name} solamente tiene "
                f"{len(image_files)} imágenes."
            )

            selected_images = image_files

        else:

            selected_images = random.sample(
                image_files,
                IMAGES_PER_CLASS
            )

        # Directorios de salida
        class_output = OUTPUT_PATH / class_name
        comparison_output = class_output / "comparison"

        class_output.mkdir(
            parents=True,
            exist_ok=True
        )

        comparison_output.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"\n{class_name}: "
            f"{len(selected_images)} imágenes"
        )

        # ----------------------------------------------------
        # Procesar imágenes
        # ----------------------------------------------------

        for index, image_path in enumerate(selected_images, start=1):

            try:

                # Abrir imagen
                with Image.open(image_path) as image:

                    original = image.convert("RGB")

                # Transformar
                transformed = transform(original)

                # Nombre de salida
                output_name = (
                    f"{index:02d}_"
                    f"{image_path.stem}.png"
                )

                output_file = (
                    class_output /
                    output_name
                )

                comparison_file = (
                    comparison_output /
                    output_name
                )

                # Guardar transformada
                transformed.save(
                    output_file
                )

                # Crear comparación
                comparison = create_comparison(
                    original,
                    transformed
                )

                comparison.save(
                    comparison_file
                )

                total_processed += 1

            except Exception as error:

                print(
                    f"  [ERROR] {image_path.name}: "
                    f"{error}"
                )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("PROCESO TERMINADO")
    print("=" * 70)

    print(
        f"\nImágenes procesadas: "
        f"{total_processed}"
    )

    print(
        f"\nResultados:"
        f"\n{OUTPUT_PATH}"
    )

    print(
        "\nDentro de cada clase encontrarás:"
        "\n  ├── imágenes transformadas"
        "\n  └── comparison/"
        "\n      └── original + transformada"
    )


if __name__ == "__main__":
    main()