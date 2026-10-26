# scripts/test_dataloader.py

import torch
from pathlib import Path
import sys
import os

ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ruta_raiz)


from datasets.dataloader import (
    get_dataloaders,
    create_datasets,
    split_samples,
    balance_samples,
    SEED,
)

from datasets.vehicle_dataset import (
    CLASS_NAMES,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
print(f"Ruta del proyecto: {PROJECT_ROOT}")
DATASET_ROOT = PROJECT_ROOT / "VehicleClassificationDataset"


BATCH_SIZE = 32
IMAGE_SIZE = 240


# ============================================================
# UTILIDADES
# ============================================================

def get_class_distribution(samples):
    """
    Obtiene la cantidad de imágenes por clase.
    """

    distribution = {
        class_idx: 0
        for class_idx in range(len(CLASS_NAMES))
    }

    for _, label in samples:
        distribution[label] += 1

    return distribution


# ============================================================
# TEST 1 — DATASET ORIGINAL
# ============================================================

def test_original_dataset():

    print("\n" + "=" * 70)
    print("TEST 1 — DATASET ORIGINAL")
    print("=" * 70)

    train_dataset, val_dataset, test_dataset = create_datasets(
        root_dir=DATASET_ROOT,
        image_size=IMAGE_SIZE,
        seed=SEED,
    )

    total = (
        len(train_dataset)
        + len(val_dataset)
        + len(test_dataset)
    )

    print(f"Total después del balanceo: {total:,}")

    assert len(CLASS_NAMES) == 10, (
        f"Se esperaban 10 clases, "
        f"pero se encontraron {len(CLASS_NAMES)}."
    )

    print("✓ 10 clases configuradas correctamente")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# TEST 2 — BALANCEO
# ============================================================

def test_balancing(train_dataset, val_dataset, test_dataset):

    print("\n" + "=" * 70)
    print("TEST 2 — BALANCEO")
    print("=" * 70)

    all_samples = (
        train_dataset.samples
        + val_dataset.samples
        + test_dataset.samples
    )

    distribution = get_class_distribution(
        all_samples
    )

    print("\nDistribución balanceada:")

    for class_idx, class_name in enumerate(CLASS_NAMES):

        count = distribution[class_idx]

        print(
            f"{class_idx:2d} | "
            f"{class_name:<25} | "
            f"{count:5d}"
        )

    counts = list(distribution.values())

    assert len(set(counts)) == 1, (
        "El dataset NO está balanceado."
    )

    print("\n✓ Todas las clases tienen el mismo número de imágenes")


# ============================================================
# TEST 3 — SPLITS
# ============================================================

def test_split_distribution(
    train_dataset,
    val_dataset,
    test_dataset,
):

    print("\n" + "=" * 70)
    print("TEST 3 — SPLIT 70 / 15 / 15")
    print("=" * 70)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    total = train_size + val_size + test_size

    train_ratio = train_size / total
    val_ratio = val_size / total
    test_ratio = test_size / total

    print(
        f"Train: {train_size:,} "
        f"({train_ratio * 100:.2f}%)"
    )

    print(
        f"Val:   {val_size:,} "
        f"({val_ratio * 100:.2f}%)"
    )

    print(
        f"Test:  {test_size:,} "
        f"({test_ratio * 100:.2f}%)"
    )

    # Permitimos una pequeña diferencia debido
    # a que el número total de imágenes puede
    # no ser divisible exactamente.

    assert abs(train_ratio - 0.70) < 0.01
    assert abs(val_ratio - 0.15) < 0.01
    assert abs(test_ratio - 0.15) < 0.01

    print("\n✓ Split 70 / 15 / 15 correcto")


# ============================================================
# TEST 4 — TODAS LAS CLASES EN CADA SPLIT
# ============================================================

def test_all_classes_present(
    train_dataset,
    val_dataset,
    test_dataset,
):

    print("\n" + "=" * 70)
    print("TEST 4 — CLASES EN CADA SPLIT")
    print("=" * 70)

    datasets = {
        "Train": train_dataset,
        "Validation": val_dataset,
        "Test": test_dataset,
    }

    for split_name, dataset in datasets.items():

        labels = set(dataset.labels)

        expected = set(
            range(len(CLASS_NAMES))
        )

        assert labels == expected, (
            f"{split_name} no contiene todas las clases."
        )

        print(
            f"✓ {split_name}: "
            f"{len(labels)} clases presentes"
        )


# ============================================================
# TEST 5 — DATALOADER
# ============================================================

def test_dataloaders():

    print("\n" + "=" * 70)
    print("TEST 5 — DATALOADERS")
    print("=" * 70)

    (
        train_loader,
        val_loader,
        test_loader,
    ) = get_dataloaders(
        root_dir=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        seed=SEED,
    )

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Val batches:   {len(val_loader)}"
    )

    print(
        f"Test batches:  {len(test_loader)}"
    )

    # --------------------------------------------------------
    # Obtener un batch
    # --------------------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    print("\nBatch obtenido:")

    print(
        f"Images shape: {images.shape}"
    )

    print(
        f"Labels shape: {labels.shape}"
    )

    # --------------------------------------------------------
    # Verificar dimensiones
    # --------------------------------------------------------

    expected_image_shape = (
        BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert images.shape == expected_image_shape, (
        f"Shape incorrecto.\n"
        f"Esperado: {expected_image_shape}\n"
        f"Obtenido: {images.shape}"
    )

    print(
        "✓ Shape de imágenes correcto: "
        f"{tuple(images.shape)}"
    )

    # --------------------------------------------------------
    # Verificar labels
    # --------------------------------------------------------

    assert labels.min() >= 0
    assert labels.max() < len(CLASS_NAMES)

    print(
        f"✓ Labels válidos: "
        f"{labels.min().item()} - "
        f"{labels.max().item()}"
    )

    # --------------------------------------------------------
    # Verificar tipo
    # --------------------------------------------------------

    assert images.dtype == torch.float32

    assert labels.dtype == torch.int64

    print("✓ Tipo de imágenes: float32")
    print("✓ Tipo de labels: int64")

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# TEST 6 — TRANSFORMS
# ============================================================

def test_transforms(train_loader):

    print("\n" + "=" * 70)
    print("TEST 6 — TRANSFORMS")
    print("=" * 70)

    images, _ = next(
        iter(train_loader)
    )

    # Después de Normalize, los valores ya no
    # están en el rango [0, 1].

    print(
        f"Valor mínimo: {images.min().item():.4f}"
    )

    print(
        f"Valor máximo: {images.max().item():.4f}"
    )

    print(
        f"Media del batch: {images.mean().item():.4f}"
    )

    print(
        f"Std del batch: {images.std().item():.4f}"
    )

    assert images.ndim == 4

    assert images.shape[1] == 3

    print(
        "\n✓ Transformaciones aplicadas correctamente"
    )


# ============================================================
# TEST 7 — REPRODUCIBILIDAD DEL SPLIT
# ============================================================

def test_reproducibility():

    print("\n" + "=" * 70)
    print("TEST 7 — REPRODUCIBILIDAD")
    print("=" * 70)

    dataset_1 = create_datasets(
        root_dir=DATASET_ROOT,
        image_size=IMAGE_SIZE,
        seed=SEED,
    )

    dataset_2 = create_datasets(
        root_dir=DATASET_ROOT,
        image_size=IMAGE_SIZE,
        seed=SEED,
    )

    for ds1, ds2 in zip(
        dataset_1,
        dataset_2,
    ):

        paths_1 = [
            str(path)
            for path, _ in ds1.samples
        ]

        paths_2 = [
            str(path)
            for path, _ in ds2.samples
        ]

        assert paths_1 == paths_2

    print(
        "✓ Mismo seed → mismo split"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("      TEST COMPLETO DEL PIPELINE DE DATOS")
    print("=" * 70)

    # --------------------------------------------------------
    # Crear datasets
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = test_original_dataset()

    # --------------------------------------------------------
    # Balanceo
    # --------------------------------------------------------

    test_balancing(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    test_split_distribution(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    # --------------------------------------------------------
    # Clases
    # --------------------------------------------------------

    test_all_classes_present(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = test_dataloaders()

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    test_transforms(
        train_loader
    )

    # --------------------------------------------------------
    # Reproducibilidad
    # --------------------------------------------------------

    test_reproducibility()

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("              TODOS LOS TESTS PASARON")
    print("=" * 70)
    print("\nEl pipeline de datos está listo para los modelos.\n")