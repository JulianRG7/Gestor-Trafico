from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Ruta raíz del dataset de clasificación
# Ejemplo:
# DATASET_ROOT/
# ├── Ambulance/
# ├── Car/
# ├── Cargo/
# └── ...
DATASET_ROOT = Path(
     "/home/julianrgx/Escritorio/UG/LIDIA_8vo_Semestre/Proyecto_Integrador/VehicleClassificationDataset"
)

# Carpeta donde se guardará todo el análisis
ANALYSIS_DIR = DATASET_ROOT / "analysis"

PLOTS_DIR = ANALYSIS_DIR / "plots"


# ============================================================
# EXTENSIONES VÁLIDAS
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# UMBRALES PARA DETECTAR CASOS SOSPECHOSOS
# ============================================================

# IMPORTANTE:
# Estos valores NO eliminan imágenes.
# Solamente sirven para marcarlas para revisión.

MIN_WIDTH = 20
MIN_HEIGHT = 5

EXTREME_ASPECT_RATIO_HIGH = 5.0
EXTREME_ASPECT_RATIO_LOW = 0.20


# ============================================================
# PERCENTILES
# ============================================================

PERCENTILES = [
    1,
    5,
    25,
    50,
    75,
    95,
    99
]


# ============================================================
# CREAR DIRECTORIOS
# ============================================================

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def get_percentiles(values):
    """
    Calcula los percentiles solicitados.

    Devuelve un diccionario.
    """

    if len(values) == 0:
        return {
            f"p{p}": np.nan
            for p in PERCENTILES
        }

    result = {}

    for p in PERCENTILES:
        result[f"p{p}"] = np.percentile(
            values,
            p
        )

    return result


def format_float(value):
    """
    Formatea números para el reporte de texto.
    """

    if pd.isna(value):
        return "N/A"

    return f"{value:.4f}"


# ============================================================
# ANALIZAR IMAGEN
# ============================================================

def analyze_image(
    image_path,
    class_name
):
    """
    Analiza una imagen individual.

    Devuelve un diccionario con:
        clase
        ruta
        nombre
        width
        height
        aspect_ratio
        status
    """

    try:

        with Image.open(image_path) as image:

            width, height = image.size

            # Verificación adicional
            image.verify()

    except Exception as e:

        return {
            "class": class_name,
            "image": image_path.name,
            "path": str(image_path),
            "width": np.nan,
            "height": np.nan,
            "aspect_ratio": np.nan,
            "status": "corrupted",
            "error": str(e)
        }

    # --------------------------------------------------------
    # Dimensiones inválidas
    # --------------------------------------------------------

    if width <= 0 or height <= 0:

        return {
            "class": class_name,
            "image": image_path.name,
            "path": str(image_path),
            "width": width,
            "height": height,
            "aspect_ratio": np.nan,
            "status": "invalid_dimensions",
            "error": "Width or height <= 0"
        }

    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    aspect_ratio = width / height

    # --------------------------------------------------------
    # Detectar casos sospechosos
    # --------------------------------------------------------

    reasons = []

    if width < MIN_WIDTH:
        reasons.append(
            f"width<{MIN_WIDTH}"
        )

    if height < MIN_HEIGHT:
        reasons.append(
            f"height<{MIN_HEIGHT}"
        )

    if aspect_ratio > EXTREME_ASPECT_RATIO_HIGH:
        reasons.append(
            f"aspect_ratio>{EXTREME_ASPECT_RATIO_HIGH}"
        )

    if aspect_ratio < EXTREME_ASPECT_RATIO_LOW:
        reasons.append(
            f"aspect_ratio<{EXTREME_ASPECT_RATIO_LOW}"
        )

    if reasons:

        status = "suspicious"

        error = "; ".join(reasons)

    else:

        status = "valid"

        error = ""

    return {
        "class": class_name,
        "image": image_path.name,
        "path": str(image_path),
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "status": status,
        "error": error
    }


# ============================================================
# BUSCAR DATASET
# ============================================================

def collect_images():

    records = []

    print()
    print("=" * 70)
    print("ANALIZANDO DATASET")
    print("=" * 70)
    print()

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"No existe el dataset:\n{DATASET_ROOT}"
        )

    # --------------------------------------------------------
    # Cada subcarpeta representa una clase
    # --------------------------------------------------------

    class_dirs = sorted(
        [
            p
            for p in DATASET_ROOT.iterdir()
            if p.is_dir()
            and p.name != "analysis"
        ]
    )

    if not class_dirs:

        raise RuntimeError(
            "No se encontraron carpetas de clases."
        )

    print(
        f"Dataset: {DATASET_ROOT}"
    )

    print(
        f"Clases encontradas: {len(class_dirs)}"
    )

    print()

    # --------------------------------------------------------
    # Procesar cada clase
    # --------------------------------------------------------

    for class_dir in class_dirs:

        class_name = class_dir.name

        image_paths = sorted(
            [
                p
                for p in class_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower()
                in VALID_EXTENSIONS
            ]
        )

        print(
            f"[{class_name}] "
            f"{len(image_paths)} imágenes"
        )

        for image_path in image_paths:

            record = analyze_image(
                image_path,
                class_name
            )

            records.append(record)

    return records


# ============================================================
# DATAFRAME
# ============================================================

def create_dataframe(records):

    df = pd.DataFrame(records)

    if df.empty:

        raise RuntimeError(
            "No se encontraron imágenes."
        )

    return df


# ============================================================
# ESTADÍSTICAS POR CLASE
# ============================================================

def generate_class_statistics(df):

    valid_df = df[
        df["status"] != "corrupted"
    ].copy()

    rows = []

    for class_name in sorted(
        valid_df["class"].unique()
    ):

        class_df = valid_df[
            valid_df["class"] == class_name
        ]

        widths = class_df[
            "width"
        ].dropna().values

        heights = class_df[
            "height"
        ].dropna().values

        ratios = class_df[
            "aspect_ratio"
        ].dropna().values

        row = {
            "class": class_name,
            "images": len(class_df),

            "width_min": np.min(widths),
            "width_mean": np.mean(widths),
            "width_median": np.median(widths),
            "width_max": np.max(widths),

            "height_min": np.min(heights),
            "height_mean": np.mean(heights),
            "height_median": np.median(heights),
            "height_max": np.max(heights),

            "aspect_ratio_min": np.min(ratios),
            "aspect_ratio_mean": np.mean(ratios),
            "aspect_ratio_median": np.median(ratios),
            "aspect_ratio_max": np.max(ratios)
        }

        # ----------------------------------------------------
        # Percentiles
        # ----------------------------------------------------

        width_percentiles = get_percentiles(
            widths
        )

        height_percentiles = get_percentiles(
            heights
        )

        ratio_percentiles = get_percentiles(
            ratios
        )

        for key, value in width_percentiles.items():

            row[
                f"width_{key}"
            ] = value

        for key, value in height_percentiles.items():

            row[
                f"height_{key}"
            ] = value

        for key, value in ratio_percentiles.items():

            row[
                f"aspect_ratio_{key}"
            ] = value

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# ESTADÍSTICAS GLOBALES
# ============================================================

def generate_global_statistics(df):

    valid_df = df[
        df["status"] != "corrupted"
    ].copy()

    widths = valid_df[
        "width"
    ].dropna().values

    heights = valid_df[
        "height"
    ].dropna().values

    ratios = valid_df[
        "aspect_ratio"
    ].dropna().values

    result = {
        "total_images": len(df),

        "valid_images": int(
            (df["status"] == "valid").sum()
        ),

        "suspicious_images": int(
            (df["status"] == "suspicious").sum()
        ),

        "corrupted_images": int(
            (df["status"] == "corrupted").sum()
        ),

        "invalid_dimension_images": int(
            (df["status"] == "invalid_dimensions").sum()
        ),

        "width_min": np.min(widths),
        "width_mean": np.mean(widths),
        "width_median": np.median(widths),
        "width_max": np.max(widths),

        "height_min": np.min(heights),
        "height_mean": np.mean(heights),
        "height_median": np.median(heights),
        "height_max": np.max(heights),

        "aspect_ratio_min": np.min(ratios),
        "aspect_ratio_mean": np.mean(ratios),
        "aspect_ratio_median": np.median(ratios),
        "aspect_ratio_max": np.max(ratios)
    }

    # --------------------------------------------------------
    # Percentiles globales
    # --------------------------------------------------------

    for key, value in get_percentiles(
        widths
    ).items():

        result[
            f"width_{key}"
        ] = value

    for key, value in get_percentiles(
        heights
    ).items():

        result[
            f"height_{key}"
        ] = value

    for key, value in get_percentiles(
        ratios
    ).items():

        result[
            f"aspect_ratio_{key}"
        ] = value

    return result


# ============================================================
# GUARDAR CASOS EXTREMOS
# ============================================================

def save_extreme_cases(df):

    # --------------------------------------------------------
    # Dimensiones pequeñas
    # --------------------------------------------------------

    extreme_dimensions = df[
        (
            (df["width"] < MIN_WIDTH)
            |
            (df["height"] < MIN_HEIGHT)
        )
        &
        (df["status"] != "corrupted")
    ].copy()

    extreme_dimensions = extreme_dimensions.sort_values(
        by=["height", "width"]
    )

    dimensions_path = (
        ANALYSIS_DIR
        / "extreme_dimensions.csv"
    )

    extreme_dimensions.to_csv(
        dimensions_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Aspect ratios extremos
    # --------------------------------------------------------

    extreme_ratios = df[
        (
            (df["aspect_ratio"] > EXTREME_ASPECT_RATIO_HIGH)
            |
            (df["aspect_ratio"] < EXTREME_ASPECT_RATIO_LOW)
        )
        &
        (df["status"] != "corrupted")
    ].copy()

    extreme_ratios = extreme_ratios.sort_values(
        by="aspect_ratio"
    )

    ratios_path = (
        ANALYSIS_DIR
        / "extreme_aspect_ratios.csv"
    )

    extreme_ratios.to_csv(
        ratios_path,
        index=False,
        encoding="utf-8"
    )

    return (
        extreme_dimensions,
        extreme_ratios
    )


# ============================================================
# GUARDAR IMÁGENES CORRUPTAS
# ============================================================

def save_corrupted_images(df):

    corrupted = df[
        df["status"] == "corrupted"
    ].copy()

    path = (
        ANALYSIS_DIR
        / "corrupted_images.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for _, row in corrupted.iterrows():

            f.write(
                f"{row['path']}\n"
            )

            f.write(
                f"ERROR: {row['error']}\n"
            )

            f.write("\n")

    return corrupted


# ============================================================
# DISTRIBUCIÓN POR CLASE
# ============================================================

def plot_class_distribution(df):

    counts = (
        df["class"]
        .value_counts()
        .sort_index()
    )

    plt.figure(
        figsize=(12, 6)
    )

    counts.plot(
        kind="bar"
    )

    plt.title(
        "Distribución de imágenes por clase"
    )

    plt.xlabel(
        "Clase"
    )

    plt.ylabel(
        "Número de imágenes"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "class_distribution.png",
        dpi=200
    )

    plt.close()


# ============================================================
# HISTOGRAMA DE WIDTH
# ============================================================

def plot_width_distribution(df):

    values = df[
        df["width"].notna()
    ]["width"]

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        values,
        bins=50
    )

    plt.title(
        "Distribución del ancho de los crops"
    )

    plt.xlabel(
        "Width (pixels)"
    )

    plt.ylabel(
        "Número de imágenes"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "width_distribution.png",
        dpi=200
    )

    plt.close()


# ============================================================
# HISTOGRAMA DE HEIGHT
# ============================================================

def plot_height_distribution(df):

    values = df[
        df["height"].notna()
    ]["height"]

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        values,
        bins=50
    )

    plt.title(
        "Distribución del alto de los crops"
    )

    plt.xlabel(
        "Height (pixels)"
    )

    plt.ylabel(
        "Número de imágenes"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "height_distribution.png",
        dpi=200
    )

    plt.close()


# ============================================================
# HISTOGRAMA DE ASPECT RATIO
# ============================================================

def plot_aspect_ratio_distribution(df):

    values = df[
        df["aspect_ratio"].notna()
    ]["aspect_ratio"]

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        values,
        bins=100
    )

    plt.title(
        "Distribución del aspect ratio"
    )

    plt.xlabel(
        "Width / Height"
    )

    plt.ylabel(
        "Número de imágenes"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / "aspect_ratio_distribution.png",
        dpi=200
    )

    plt.close()


# ============================================================
# REPORTE DE TEXTO
# ============================================================

def generate_report(
    df,
    class_stats,
    global_stats,
    extreme_dimensions,
    extreme_ratios
):

    report_path = (
        ANALYSIS_DIR
        / "dataset_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "DATASET ANALYSIS REPORT V2\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        # ----------------------------------------------------
        # RESUMEN
        # ----------------------------------------------------

        f.write(
            "RESUMEN\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Total de imágenes: "
            f"{global_stats['total_images']}\n"
        )

        f.write(
            f"Imágenes válidas: "
            f"{global_stats['valid_images']}\n"
        )

        f.write(
            f"Imágenes sospechosas: "
            f"{global_stats['suspicious_images']}\n"
        )

        f.write(
            f"Imágenes corruptas: "
            f"{global_stats['corrupted_images']}\n"
        )

        f.write(
            f"Dimensiones inválidas: "
            f"{global_stats['invalid_dimension_images']}\n"
        )

        f.write("\n")

        # ----------------------------------------------------
        # DISTRIBUCIÓN
        # ----------------------------------------------------

        f.write(
            "DISTRIBUCIÓN POR CLASE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        class_counts = (
            df["class"]
            .value_counts()
            .sort_index()
        )

        for class_name, count in class_counts.items():

            percentage = (
                count
                / len(df)
                * 100
            )

            f.write(
                f"{class_name:30s} "
                f"{count:6d} "
                f"({percentage:6.2f}%)\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # ESTADÍSTICAS GLOBALES
        # ----------------------------------------------------

        f.write(
            "ESTADÍSTICAS GLOBALES\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for key, value in global_stats.items():

            if isinstance(
                value,
                (float, np.floating)
            ):

                f.write(
                    f"{key:30s}: "
                    f"{value:.4f}\n"
                )

            else:

                f.write(
                    f"{key:30s}: "
                    f"{value}\n"
                )

        f.write("\n")

        # ----------------------------------------------------
        # ESTADÍSTICAS POR CLASE
        # ----------------------------------------------------

        f.write(
            "ESTADÍSTICAS POR CLASE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for _, row in class_stats.iterrows():

            f.write(
                f"\n[{row['class']}]\n"
            )

            f.write(
                f"Images: "
                f"{int(row['images'])}\n"
            )

            f.write(
                f"Width: "
                f"min={row['width_min']:.2f}, "
                f"median={row['width_median']:.2f}, "
                f"max={row['width_max']:.2f}\n"
            )

            f.write(
                f"Height: "
                f"min={row['height_min']:.2f}, "
                f"median={row['height_median']:.2f}, "
                f"max={row['height_max']:.2f}\n"
            )

            f.write(
                f"Aspect ratio: "
                f"min={row['aspect_ratio_min']:.4f}, "
                f"median={row['aspect_ratio_median']:.4f}, "
                f"max={row['aspect_ratio_max']:.4f}\n"
            )

            f.write(
                "Width percentiles: "
            )

            for p in PERCENTILES:

                f.write(
                    f"P{p}="
                    f"{row[f'width_p{p}']:.2f} "
                )

            f.write("\n")

            f.write(
                "Height percentiles: "
            )

            for p in PERCENTILES:

                f.write(
                    f"P{p}="
                    f"{row[f'height_p{p}']:.2f} "
                )

            f.write("\n")

            f.write(
                "Aspect ratio percentiles: "
            )

            for p in PERCENTILES:

                f.write(
                    f"P{p}="
                    f"{row[f'aspect_ratio_p{p}']:.4f} "
                )

            f.write("\n")

        # ----------------------------------------------------
        # CASOS EXTREMOS
        # ----------------------------------------------------

        f.write("\n")

        f.write(
            "CASOS EXTREMOS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Dimensiones sospechosas: "
            f"{len(extreme_dimensions)}\n"
        )

        f.write(
            f"Aspect ratios extremos: "
            f"{len(extreme_ratios)}\n"
        )

        f.write("\n")

        f.write(
            "IMPORTANTE:\n"
        )

        f.write(
            "Las imágenes sospechosas NO fueron eliminadas.\n"
        )

        f.write(
            "Deben revisarse visualmente antes de decidir "
            "si son errores de extracción o casos válidos.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DATASET ANALYSIS V2")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # 1. Analizar imágenes
    # --------------------------------------------------------

    records = collect_images()

    # --------------------------------------------------------
    # 2. Crear DataFrame
    # --------------------------------------------------------

    df = create_dataframe(
        records
    )

    # --------------------------------------------------------
    # 3. Guardar estadísticas completas
    # --------------------------------------------------------

    statistics_path = (
        ANALYSIS_DIR
        / "dataset_statistics_v2.csv"
    )

    df.to_csv(
        statistics_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 4. Estadísticas por clase
    # --------------------------------------------------------

    class_stats = generate_class_statistics(
        df
    )

    class_stats_path = (
        ANALYSIS_DIR
        / "class_statistics_v2.csv"
    )

    class_stats.to_csv(
        class_stats_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 5. Estadísticas globales
    # --------------------------------------------------------

    global_stats = generate_global_statistics(
        df
    )

    global_stats_df = pd.DataFrame(
        [global_stats]
    )

    global_stats_path = (
        ANALYSIS_DIR
        / "global_statistics_v2.csv"
    )

    global_stats_df.to_csv(
        global_stats_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 6. Casos extremos
    # --------------------------------------------------------

    (
        extreme_dimensions,
        extreme_ratios
    ) = save_extreme_cases(
        df
    )

    # --------------------------------------------------------
    # 7. Corruptas
    # --------------------------------------------------------

    corrupted = save_corrupted_images(
        df
    )

    # --------------------------------------------------------
    # 8. Gráficas
    # --------------------------------------------------------

    plot_class_distribution(
        df
    )

    plot_width_distribution(
        df
    )

    plot_height_distribution(
        df
    )

    plot_aspect_ratio_distribution(
        df
    )

    # --------------------------------------------------------
    # 9. Reporte
    # --------------------------------------------------------

    generate_report(
        df,
        class_stats,
        global_stats,
        extreme_dimensions,
        extreme_ratios
    )

    # --------------------------------------------------------
    # 10. Resumen consola
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANÁLISIS TERMINADO")
    print("=" * 70)

    print()
    print(
        f"Total imágenes: "
        f"{global_stats['total_images']}"
    )

    print(
        f"Válidas: "
        f"{global_stats['valid_images']}"
    )

    print(
        f"Sospechosas: "
        f"{global_stats['suspicious_images']}"
    )

    print(
        f"Corruptas: "
        f"{global_stats['corrupted_images']}"
    )

    print()
    print(
        f"Dimensiones sospechosas: "
        f"{len(extreme_dimensions)}"
    )

    print(
        f"Aspect ratios extremos: "
        f"{len(extreme_ratios)}"
    )

    print()
    print(
        "Archivos generados:"
    )

    print(
        f"  {statistics_path}"
    )

    print(
        f"  {class_stats_path}"
    )

    print(
        f"  {global_stats_path}"
    )

    print(
        f"  {ANALYSIS_DIR / 'extreme_dimensions.csv'}"
    )

    print(
        f"  {ANALYSIS_DIR / 'extreme_aspect_ratios.csv'}"
    )

    print(
        f"  {ANALYSIS_DIR / 'corrupted_images.txt'}"
    )

    print(
        f"  {ANALYSIS_DIR / 'dataset_report.txt'}"
    )

    print(
        f"  {PLOTS_DIR}"
    )

    print()


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()


# from pathlib import Path
# from PIL import Image
# from collections import defaultdict
# import statistics
# import csv


# # ============================================================
# # CONFIGURACIÓN
# # ============================================================

# # CAMBIA ESTA RUTA POR LA UBICACIÓN DE TU DATASET
# DATASET_PATH = Path(
#     "/home/julianrgx/Escritorio/UG/LIDIA_8vo_Semestre/Proyecto_Integrador/VehicleClassificationDataset"
# )

# OUTPUT_DIR = Path("analysis")

# IMAGE_EXTENSIONS = {
#     ".jpg",
#     ".jpeg",
#     ".png",
#     ".bmp",
#     ".webp"
# }


# # ============================================================
# # FUNCIONES
# # ============================================================

# def get_image_files(class_dir):
#     """
#     Obtiene todas las imágenes válidas por extensión
#     dentro de una carpeta de clase.
#     """

#     return [
#         path
#         for path in class_dir.iterdir()
#         if path.is_file()
#         and path.suffix.lower() in IMAGE_EXTENSIONS
#     ]


# def analyze_image(image_path):
#     """
#     Intenta abrir una imagen y obtiene:
#         width
#         height
#         aspect ratio

#     Devuelve None si la imagen está corrupta.
#     """

#     try:
#         with Image.open(image_path) as image:
#             image.verify()

#         with Image.open(image_path) as image:
#             width, height = image.size

#         if height == 0:
#             return None

#         aspect_ratio = width / height

#         return {
#             "width": width,
#             "height": height,
#             "aspect_ratio": aspect_ratio
#         }

#     except Exception:
#         return None


# def get_statistics(values):
#     """
#     Calcula estadísticas básicas.
#     """

#     if not values:
#         return {
#             "min": None,
#             "max": None,
#             "mean": None,
#             "median": None
#         }

#     return {
#         "min": min(values),
#         "max": max(values),
#         "mean": statistics.mean(values),
#         "median": statistics.median(values)
#     }


# # ============================================================
# # ANÁLISIS
# # ============================================================

# def main():

#     if not DATASET_PATH.exists():
#         print(f"[ERROR] No existe el dataset:")
#         print(DATASET_PATH)
#         return

#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#     class_dirs = sorted([
#         directory
#         for directory in DATASET_PATH.iterdir()
#         if directory.is_dir()
#     ])

#     if not class_dirs:
#         print("[ERROR] No se encontraron carpetas de clases.")
#         return

#     print("=" * 70)
#     print("ANÁLISIS DEL DATASET")
#     print("=" * 70)

#     print(f"\nDataset:")
#     print(DATASET_PATH)

#     print(f"\nNúmero de clases: {len(class_dirs)}")

#     # --------------------------------------------------------
#     # Estructuras para estadísticas
#     # --------------------------------------------------------

#     global_widths = []
#     global_heights = []
#     global_ratios = []

#     corrupted_images = []

#     class_statistics = {}

#     total_images = 0

#     # --------------------------------------------------------
#     # Recorrer clases
#     # --------------------------------------------------------

#     for class_dir in class_dirs:

#         class_name = class_dir.name

#         image_files = get_image_files(class_dir)

#         widths = []
#         heights = []
#         ratios = []

#         valid_images = 0

#         for image_path in image_files:

#             result = analyze_image(image_path)

#             if result is None:
#                 corrupted_images.append(str(image_path))
#                 continue

#             valid_images += 1

#             width = result["width"]
#             height = result["height"]
#             ratio = result["aspect_ratio"]

#             widths.append(width)
#             heights.append(height)
#             ratios.append(ratio)

#             global_widths.append(width)
#             global_heights.append(height)
#             global_ratios.append(ratio)

#         total_images += valid_images

#         class_statistics[class_name] = {
#             "total_files": len(image_files),
#             "valid_images": valid_images,
#             "corrupted_images": len(image_files) - valid_images,
#             "width": get_statistics(widths),
#             "height": get_statistics(heights),
#             "aspect_ratio": get_statistics(ratios)
#         }

#     # ========================================================
#     # MOSTRAR RESULTADOS
#     # ========================================================

#     print("\n" + "=" * 70)
#     print("IMÁGENES POR CLASE")
#     print("=" * 70)

#     for class_name, data in class_statistics.items():

#         print(
#             f"{class_name:<28}"
#             f"{data['valid_images']:>7} imágenes"
#         )

#     print("\n" + "=" * 70)
#     print("ESTADÍSTICAS GLOBALES")
#     print("=" * 70)

#     print(f"\nTotal de imágenes válidas: {total_images}")
#     print(f"Imágenes corruptas:        {len(corrupted_images)}")

#     width_stats = get_statistics(global_widths)
#     height_stats = get_statistics(global_heights)
#     ratio_stats = get_statistics(global_ratios)

#     print("\nANCHO")
#     print(f"  Mínimo:   {width_stats['min']}")
#     print(f"  Máximo:   {width_stats['max']}")
#     print(f"  Promedio: {width_stats['mean']:.2f}")
#     print(f"  Mediana:  {width_stats['median']:.2f}")

#     print("\nALTO")
#     print(f"  Mínimo:   {height_stats['min']}")
#     print(f"  Máximo:   {height_stats['max']}")
#     print(f"  Promedio: {height_stats['mean']:.2f}")
#     print(f"  Mediana:  {height_stats['median']:.2f}")

#     print("\nASPECT RATIO (width / height)")
#     print(f"  Mínimo:   {ratio_stats['min']:.4f}")
#     print(f"  Máximo:   {ratio_stats['max']:.4f}")
#     print(f"  Promedio: {ratio_stats['mean']:.4f}")
#     print(f"  Mediana:  {ratio_stats['median']:.4f}")

#     # ========================================================
#     # GUARDAR CSV
#     # ========================================================

#     csv_path = OUTPUT_DIR / "dataset_statistics.csv"

#     with open(csv_path, "w", newline="", encoding="utf-8") as file:

#         writer = csv.writer(file)

#         writer.writerow([
#             "class",
#             "total_files",
#             "valid_images",
#             "corrupted_images",
#             "width_min",
#             "width_max",
#             "width_mean",
#             "width_median",
#             "height_min",
#             "height_max",
#             "height_mean",
#             "height_median",
#             "ratio_min",
#             "ratio_max",
#             "ratio_mean",
#             "ratio_median"
#         ])

#         for class_name, data in class_statistics.items():

#             writer.writerow([
#                 class_name,
#                 data["total_files"],
#                 data["valid_images"],
#                 data["corrupted_images"],

#                 data["width"]["min"],
#                 data["width"]["max"],
#                 data["width"]["mean"],
#                 data["width"]["median"],

#                 data["height"]["min"],
#                 data["height"]["max"],
#                 data["height"]["mean"],
#                 data["height"]["median"],

#                 data["aspect_ratio"]["min"],
#                 data["aspect_ratio"]["max"],
#                 data["aspect_ratio"]["mean"],
#                 data["aspect_ratio"]["median"]
#             ])

#     # ========================================================
#     # GUARDAR IMÁGENES CORRUPTAS
#     # ========================================================

#     corrupted_path = OUTPUT_DIR / "corrupted_images.txt"

#     with open(corrupted_path, "w", encoding="utf-8") as file:

#         for image in corrupted_images:
#             file.write(image + "\n")

#     # ========================================================
#     # GUARDAR REPORTE
#     # ========================================================

#     report_path = OUTPUT_DIR / "dataset_report.txt"

#     with open(report_path, "w", encoding="utf-8") as file:

#         file.write("DATASET ANALYSIS REPORT\n")
#         file.write("=" * 70 + "\n\n")

#         file.write(f"Dataset: {DATASET_PATH}\n")
#         file.write(f"Classes: {len(class_dirs)}\n")
#         file.write(f"Valid images: {total_images}\n")
#         file.write(f"Corrupted images: {len(corrupted_images)}\n\n")

#         file.write("IMAGES PER CLASS\n")
#         file.write("-" * 70 + "\n")

#         for class_name, data in class_statistics.items():

#             file.write(
#                 f"{class_name:<30} "
#                 f"{data['valid_images']} images\n"
#             )

#         file.write("\nGLOBAL STATISTICS\n")
#         file.write("-" * 70 + "\n")

#         file.write(
#             f"Width: "
#             f"{width_stats['min']} - "
#             f"{width_stats['max']} "
#             f"(mean={width_stats['mean']:.2f}, "
#             f"median={width_stats['median']:.2f})\n"
#         )

#         file.write(
#             f"Height: "
#             f"{height_stats['min']} - "
#             f"{height_stats['max']} "
#             f"(mean={height_stats['mean']:.2f}, "
#             f"median={height_stats['median']:.2f})\n"
#         )

#         file.write(
#             f"Aspect ratio: "
#             f"{ratio_stats['min']:.4f} - "
#             f"{ratio_stats['max']:.4f} "
#             f"(mean={ratio_stats['mean']:.4f}, "
#             f"median={ratio_stats['median']:.4f})\n"
#         )

#     print("\n" + "=" * 70)
#     print("ANÁLISIS TERMINADO")
#     print("=" * 70)

#     print(f"\nResultados guardados en:")
#     print(OUTPUT_DIR.resolve())

#     print(f"\nCSV:")
#     print(csv_path.resolve())

#     print(f"\nReporte:")
#     print(report_path.resolve())

#     if corrupted_images:
#         print(f"\n[WARNING] Se encontraron "
#               f"{len(corrupted_images)} imágenes corruptas.")
#         print(f"Lista:")
#         print(corrupted_path.resolve())


# if __name__ == "__main__":
#     main()