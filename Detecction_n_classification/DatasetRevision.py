from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import itertools

import pandas as pd
import imagehash
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_SOURCES = [
    Path("YOLODataset-Camera_Free-V1"),
    Path("YOLODataset-Camera_Free3-L-V1"),
    Path("YOLODataset-Camera_Free3-L-V2"),
    Path("YOLODataset"),
]

# Carpeta donde se guardarán los reportes
REPORT_DIR = Path("./auditoria_dataset")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

CLASS_NAMES = {
    0: "Car",
    1: "Motorcycle",
    2: "Passenger",
    3: "Cargo",
    4: "Ambulance",
    5: "Police",
    6: "FireTruck"
}

# ------------------------------------------------------------
# Umbral de similitud perceptual
#
# imagehash usa distancia de Hamming.
#
# 0  = idénticas perceptualmente
# 1-5 = extremadamente similares
# 6-10 = similares
# >10 = probablemente diferentes
#
# Para auditoría empezaremos con 5.
# ------------------------------------------------------------

PHASH_THRESHOLD = 5


# ============================================================
# CREAR CARPETA DE REPORTES
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OBTENER IMÁGENES
# ============================================================

def get_images(source):

    images_dir = source / "images"

    if not images_dir.exists():
        return []

    return sorted([
        p
        for p in images_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


# ============================================================
# SHA-256
# ============================================================

def calculate_sha256(path):

    sha = hashlib.sha256()

    try:

        with open(path, "rb") as f:

            while True:

                data = f.read(1024 * 1024)

                if not data:
                    break

                sha.update(data)

        return sha.hexdigest()

    except Exception:
        return None


# ============================================================
# PHASH
# ============================================================

def calculate_phash(path):

    try:

        with Image.open(path) as img:

            img = img.convert("RGB")

            return str(
                imagehash.phash(img)
            )

    except Exception:
        return None


# ============================================================
# INFORMACIÓN BÁSICA DE IMAGEN
# ============================================================

def get_image_info(path):

    try:

        with Image.open(path) as img:

            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode
            }

    except Exception:

        return {
            "width": None,
            "height": None,
            "format": None,
            "mode": None
        }


# ============================================================
# VALIDAR LABEL YOLO
# ============================================================

def validate_label(label_path):

    errors = []

    try:

        with open(label_path, "r") as f:
            lines = f.readlines()

    except Exception as e:

        return [f"No se pudo leer: {e}"]

    if len(lines) == 0:
        return ["Label vacío"]

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        values = line.split()

        if len(values) != 5:

            errors.append(
                f"Línea {line_number}: "
                f"se esperaban 5 valores, "
                f"hay {len(values)}"
            )

            continue

        try:

            class_id = int(values[0])

            x = float(values[1])
            y = float(values[2])
            w = float(values[3])
            h = float(values[4])

        except ValueError:

            errors.append(
                f"Línea {line_number}: "
                f"valores no numéricos"
            )

            continue

        if class_id not in CLASS_NAMES:

            errors.append(
                f"Línea {line_number}: "
                f"class_id {class_id} no existe"
            )

        if not 0 <= x <= 1:
            errors.append(
                f"Línea {line_number}: x={x}"
            )

        if not 0 <= y <= 1:
            errors.append(
                f"Línea {line_number}: y={y}"
            )

        if not 0 < w <= 1:
            errors.append(
                f"Línea {line_number}: width={w}"
            )

        if not 0 < h <= 1:
            errors.append(
                f"Línea {line_number}: height={h}"
            )

    return errors


# ============================================================
# AUDITORÍA GENERAL
# ============================================================

def audit_dataset():

    print("\n")
    print("=" * 80)
    print("AUDITORÍA DEL DATASET")
    print("=" * 80)

    records = []

    # --------------------------------------------------------
    # RECORRER LAS 4 FUENTES
    # --------------------------------------------------------

    for source_id, source in enumerate(
        DATASET_SOURCES,
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"DATASET {source_id}: {source}"
        )

        images_dir = source / "images"
        labels_dir = source / "labels"

        if not images_dir.exists():

            print(
                "ERROR: no existe images/"
            )

            continue

        if not labels_dir.exists():

            print(
                "ERROR: no existe labels/"
            )

            continue

        images = get_images(source)

        print(
            f"Imágenes: {len(images)}"
        )

        for image_path in tqdm(
            images,
            desc=f"Dataset {source_id}"
        ):

            relative_path = image_path.relative_to(
                source
            )

            label_path = (
                labels_dir
                / f"{image_path.stem}.txt"
            )

            image_info = get_image_info(
                image_path
            )

            sha256 = calculate_sha256(
                image_path
            )

            phash = calculate_phash(
                image_path
            )

            label_exists = label_path.exists()

            label_errors = []

            if label_exists:

                label_errors = validate_label(
                    label_path
                )

            # ------------------------------------------------
            # CLASES DEL LABEL
            # ------------------------------------------------

            classes = []

            if label_exists and not label_errors:

                try:

                    with open(
                        label_path,
                        "r"
                    ) as f:

                        for line in f:

                            values = (
                                line.strip()
                                .split()
                            )

                            if len(values) == 5:

                                class_id = int(
                                    values[0]
                                )

                                classes.append(
                                    class_id
                                )

                except Exception:
                    pass

            records.append({

                "source_id": source_id,

                "source": str(source),

                "image": str(image_path),

                "relative_path": str(
                    relative_path
                ),

                "filename": image_path.name,

                "label": str(label_path),

                "label_exists": label_exists,

                "label_valid": (
                    label_exists
                    and len(label_errors) == 0
                ),

                "label_errors": " | ".join(
                    label_errors
                ),

                "classes": ",".join(
                    map(str, sorted(set(classes)))
                ),

                "num_objects": len(classes),

                "width": image_info["width"],

                "height": image_info["height"],

                "format": image_info["format"],

                "sha256": sha256,

                "phash": phash
            })

    df = pd.DataFrame(records)

    return df


# ============================================================
# REPORTE BÁSICO
# ============================================================

def basic_report(df):

    print("\n")
    print("=" * 80)
    print("1. RESUMEN GENERAL")
    print("=" * 80)

    print(
        f"Total imágenes: {len(df)}"
    )

    print(
        f"Sin label: "
        f"{(~df['label_exists']).sum()}"
    )

    print(
        f"Labels inválidos: "
        f"{(
            df['label_exists']
            & ~df['label_valid']
        ).sum()}"
    )

    print("\nImágenes por fuente:")

    print(
        df.groupby("source_id")
        .size()
    )

    print("\nResoluciones:")

    print(
        df[
            ["width", "height"]
        ]
        .value_counts()
        .head(20)
    )


# ============================================================
# DUPLICADOS EXACTOS
# ============================================================

def find_exact_duplicates(df):

    print("\n")
    print("=" * 80)
    print("2. DUPLICADOS EXACTOS")
    print("=" * 80)

    duplicate_groups = (
        df[
            df["sha256"].notna()
        ]
        .groupby("sha256")
        .filter(
            lambda x: len(x) > 1
        )
    )

    if duplicate_groups.empty:

        print(
            "No se encontraron "
            "duplicados exactos."
        )

        return

    print(
        f"Se encontraron "
        f"{duplicate_groups['sha256'].nunique()} "
        f"grupos duplicados."
    )

    duplicate_groups.to_csv(
        REPORT_DIR
        / "duplicados_exactos.csv",
        index=False
    )

    for sha, group in (
        duplicate_groups
        .groupby("sha256")
    ):

        print("\nGRUPO:")

        for path in group["image"]:

            print(
                f"  {path}"
            )


# ============================================================
# IMÁGENES MUY SIMILARES
# ============================================================

def find_similar_images(df):

    print("\n")
    print("=" * 80)
    print("3. IMÁGENES CASI IDÉNTICAS")
    print("=" * 80)

    valid_df = df[
        df["phash"].notna()
    ].reset_index(drop=True)

    print(
        f"Comparando "
        f"{len(valid_df)} imágenes..."
    )

    pairs = []

    # --------------------------------------------------------
    # Comparación por pares
    # --------------------------------------------------------
    #
    # OJO:
    # Si tienes 20,000 imágenes esto puede ser costoso.
    #
    # 20,000² / 2 ≈ 200 millones de comparaciones.
    #
    # Para datasets grandes posteriormente podemos hacer
    # una estrategia más eficiente por bloques.
    # --------------------------------------------------------

    hashes = []

    for value in valid_df["phash"]:

        hashes.append(
            imagehash.hex_to_hash(value)
        )

    for i, j in tqdm(
        itertools.combinations(
            range(len(hashes)),
            2
        ),
        total=(
            len(hashes)
            * (len(hashes) - 1)
            // 2
        ),
        desc="Comparando"
    ):

        distance = (
            hashes[i] - hashes[j]
        )

        if distance <= PHASH_THRESHOLD:

            pairs.append({

                "image_1":
                    valid_df.iloc[i]["image"],

                "source_1":
                    valid_df.iloc[i]["source_id"],

                "image_2":
                    valid_df.iloc[j]["image"],

                "source_2":
                    valid_df.iloc[j]["source_id"],

                "phash_distance":
                    distance
            })

    if not pairs:

        print(
            "\nNo se encontraron "
            "imágenes casi idénticas."
        )

        return

    similar_df = pd.DataFrame(
        pairs
    )

    similar_df = similar_df.sort_values(
        "phash_distance"
    )

    similar_df.to_csv(
        REPORT_DIR
        / "imagenes_similares.csv",
        index=False
    )

    print(
        f"\nSe encontraron "
        f"{len(similar_df)} pares "
        f"similares."
    )

    print(
        "\nPrimeros resultados:"
    )

    print(
        similar_df.head(20).to_string(
            index=False
        )
    )


# ============================================================
# ANÁLISIS DE NOMBRES
# ============================================================

def analyze_filenames(df):

    print("\n")
    print("=" * 80)
    print("4. ANÁLISIS DE NOMBRES")
    print("=" * 80)

    print(
        "\nEjemplos de nombres:"
    )

    for source_id in sorted(
        df["source_id"].unique()
    ):

        subset = df[
            df["source_id"]
            == source_id
        ]

        print(
            f"\nDataset {source_id}:"
        )

        for filename in (
            subset["filename"]
            .head(10)
        ):

            print(
                f"  {filename}"
            )


# ============================================================
# DISTRIBUCIÓN DE CLASES
# ============================================================

def analyze_classes(df):

    print("\n")
    print("=" * 80)
    print("5. DISTRIBUCIÓN DE CLASES")
    print("=" * 80)

    counter = Counter()

    for _, row in df.iterrows():

        if not row["label_valid"]:
            continue

        if not row["classes"]:
            continue

        classes = row["classes"].split(",")

        for class_id in classes:

            counter[
                int(class_id)
            ] += 1

    print()

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"{class_id}: "
            f"{class_name:<15} "
            f"{counter[class_id]}"
        )

    # Guardar
    class_df = pd.DataFrame({

        "class_id":
            list(CLASS_NAMES.keys()),

        "class_name":
            list(CLASS_NAMES.values()),

        "objects":
            [
                counter[i]
                for i in CLASS_NAMES
            ]
    })

    class_df.to_csv(
        REPORT_DIR
        / "distribucion_clases.csv",
        index=False
    )


# ============================================================
# COMPARAR SIMILITUD ENTRE FUENTES
# ============================================================

def analyze_cross_source_similarity(df):

    print("\n")
    print("=" * 80)
    print("6. POSIBLE FUGA ENTRE DATASETS")
    print("=" * 80)

    valid_df = df[
        df["phash"].notna()
    ].reset_index(drop=True)

    hashes = [
        imagehash.hex_to_hash(x)
        for x in valid_df["phash"]
    ]

    cross_source_pairs = []

    for i, j in tqdm(
        itertools.combinations(
            range(len(hashes)),
            2
        ),
        total=(
            len(hashes)
            * (len(hashes) - 1)
            // 2
        ),
        desc="Comparando fuentes"
    ):

        source_i = (
            valid_df.iloc[i]["source_id"]
        )

        source_j = (
            valid_df.iloc[j]["source_id"]
        )

        # Solo interesa comparar
        # imágenes de diferentes fuentes.
        if source_i == source_j:
            continue

        distance = (
            hashes[i] - hashes[j]
        )

        if distance <= PHASH_THRESHOLD:

            cross_source_pairs.append({

                "source_1":
                    source_i,

                "image_1":
                    valid_df.iloc[i]["image"],

                "source_2":
                    source_j,

                "image_2":
                    valid_df.iloc[j]["image"],

                "phash_distance":
                    distance
            })

    if not cross_source_pairs:

        print(
            "\nNo se encontraron "
            "imágenes similares "
            "entre fuentes diferentes."
        )

        return

    result = pd.DataFrame(
        cross_source_pairs
    )

    result = result.sort_values(
        "phash_distance"
    )

    result.to_csv(
        REPORT_DIR
        / "fuga_entre_fuentes.csv",
        index=False
    )

    print(
        f"\nATENCIÓN:"
        f" se encontraron "
        f"{len(result)} pares "
        f"similares entre fuentes."
    )

    print(
        "\nPrimeros casos:"
    )

    print(
        result.head(30).to_string(
            index=False
        )
    )


# ============================================================
# GUARDAR REPORTE COMPLETO
# ============================================================

def save_full_report(df):

    path = (
        REPORT_DIR
        / "auditoria_completa.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"\nReporte completo:"
        f"\n{path.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = audit_dataset()

    if df.empty:

        print(
            "\nNo se encontraron imágenes."
        )

        return

    basic_report(df)

    find_exact_duplicates(df)

    analyze_filenames(df)

    analyze_classes(df)

    # IMPORTANTE:
    # Estas dos funciones pueden tardar bastante
    # con datasets grandes.

    find_similar_images(df)

    analyze_cross_source_similarity(df)

    save_full_report(df)

    print("\n")
    print("=" * 80)
    print("AUDITORÍA TERMINADA")
    print("=" * 80)

    print(
        f"\nTodos los reportes están en:"
        f"\n{REPORT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()