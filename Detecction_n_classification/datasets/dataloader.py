# datasets/dataloader.py
# datasets/dataloader.py

import random

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets.vehicle_dataset import (
    VehicleClassificationDataset,
    CLASS_NAMES,
)

from datasets.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 7

BATCH_SIZE = 32

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

NUM_WORKERS = 4

PIN_MEMORY = True

IMAGE_SIZE = 240

ALLOW_UPSCALE = True


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

def set_seed(seed=SEED):
    """
    Fija las semillas para hacer reproducibles los experimentos.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------
    # Reproducibilidad CUDA
    # --------------------------------------------------------

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# BALANCEO
# ============================================================

def balance_samples(
    samples,
    seed=SEED,
):
    """
    Balancea el dataset mediante undersampling.

    Se toma aleatoriamente el mismo número de muestras
    para todas las clases.

    Ese número corresponde al tamaño de la clase
    con menor cantidad de imágenes.

    Parameters
    ----------
    samples : list
        Lista de:

            (image_path, label)

    seed : int
        Semilla utilizada para la selección aleatoria.

    Returns
    -------
    balanced_samples : list
        Samples balanceados.
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Agrupar muestras por clase
    # --------------------------------------------------------

    class_samples = {
        class_idx: []
        for class_idx in range(len(CLASS_NAMES))
    }

    for sample in samples:

        image_path, label = sample

        class_samples[label].append(
            (image_path, label)
        )

    # --------------------------------------------------------
    # Distribución original
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DISTRIBUCIÓN ORIGINAL")
    print("=" * 60)

    for class_idx, class_name in enumerate(CLASS_NAMES):

        count = len(
            class_samples[class_idx]
        )

        print(
            f"{class_idx:2d} | "
            f"{class_name:<25} | "
            f"{count:5d}"
        )

    # --------------------------------------------------------
    # Encontrar clase minoritaria
    # --------------------------------------------------------

    min_samples = min(
        len(class_samples[class_idx])
        for class_idx in class_samples
    )

    print("\nClase minoritaria:")

    minority_classes = [
        CLASS_NAMES[class_idx]
        for class_idx in class_samples
        if len(class_samples[class_idx]) == min_samples
    ]

    for class_name in minority_classes:
        print(
            f"  {class_name}: {min_samples}"
        )

    # --------------------------------------------------------
    # Seleccionar muestras
    # --------------------------------------------------------

    balanced_samples = []

    for class_idx in range(len(CLASS_NAMES)):

        samples_class = class_samples[class_idx]

        # Convertimos a índices para hacer la selección
        # aleatoria sin reemplazo.
        selected_indices = rng.choice(
            len(samples_class),
            size=min_samples,
            replace=False,
        )

        for index in selected_indices:

            balanced_samples.append(
                samples_class[index]
            )

    # --------------------------------------------------------
    # Mezclar dataset completo
    # --------------------------------------------------------

    rng.shuffle(balanced_samples)

    # --------------------------------------------------------
    # Mostrar distribución final
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DISTRIBUCIÓN BALANCEADA")
    print("=" * 60)

    for class_idx, class_name in enumerate(CLASS_NAMES):

        count = sum(
            1
            for _, label in balanced_samples
            if label == class_idx
        )

        print(
            f"{class_idx:2d} | "
            f"{class_name:<25} | "
            f"{count:5d}"
        )

    print("-" * 60)
    print(
        f"Total: {len(balanced_samples)} imágenes"
    )

    return balanced_samples


# ============================================================
# SPLIT ESTRATIFICADO
# ============================================================

def split_samples(
    samples,
    seed=SEED,
):
    """
    Divide las muestras en:

        70% Train
        15% Validation
        15% Test

    manteniendo la distribución de clases.

    Parameters
    ----------
    samples : list
        Samples balanceados.

    seed : int
        Semilla.

    Returns
    -------
    train_samples
    val_samples
    test_samples
    """

    samples = np.array(
        samples,
        dtype=object,
    )

    labels = np.array([
        sample[1]
        for sample in samples
    ])

    # --------------------------------------------------------
    # TRAIN = 70%
    # TEMP = 30%
    # --------------------------------------------------------

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(
            VAL_RATIO + TEST_RATIO
        ),
        stratify=labels,
        random_state=seed,
    )

    # --------------------------------------------------------
    # Labels del conjunto temporal
    # --------------------------------------------------------

    temp_labels = np.array([
        sample[1]
        for sample in temp_samples
    ])

    # --------------------------------------------------------
    # TEMP → 50% VAL / 50% TEST
    # --------------------------------------------------------

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.5,
        stratify=temp_labels,
        random_state=seed,
    )

    return (
        train_samples.tolist(),
        val_samples.tolist(),
        test_samples.tolist(),
    )


# ============================================================
# VERIFICAR DISTRIBUCIÓN
# ============================================================

def print_split_distribution(
    train_samples,
    val_samples,
    test_samples,
):
    """
    Imprime la distribución de clases de cada split.
    """

    print("\n" + "=" * 70)
    print("DISTRIBUCIÓN DE LOS SPLITS")
    print("=" * 70)

    print(
        f"{'Clase':<25}"
        f"{'Train':>10}"
        f"{'Val':>10}"
        f"{'Test':>10}"
    )

    print("-" * 70)

    for class_idx, class_name in enumerate(CLASS_NAMES):

        train_count = sum(
            label == class_idx
            for _, label in train_samples
        )

        val_count = sum(
            label == class_idx
            for _, label in val_samples
        )

        test_count = sum(
            label == class_idx
            for _, label in test_samples
        )

        print(
            f"{class_name:<25}"
            f"{train_count:>10}"
            f"{val_count:>10}"
            f"{test_count:>10}"
        )

    print("-" * 70)

    print(
        f"{'TOTAL':<25}"
        f"{len(train_samples):>10}"
        f"{len(val_samples):>10}"
        f"{len(test_samples):>10}"
    )


# ============================================================
# CREAR DATASETS
# ============================================================

def create_datasets(
    root_dir,
    image_size=IMAGE_SIZE,
    allow_upscale=ALLOW_UPSCALE,
    seed=SEED,
):
    """
    Crea los datasets balanceados de:

        Train
        Validation
        Test

    """

    # --------------------------------------------------------
    # Fijar semilla
    # --------------------------------------------------------

    set_seed(seed)

    # --------------------------------------------------------
    # Dataset base
    #
    # Sin transform porque solamente necesitamos obtener
    # las muestras y sus labels.
    # --------------------------------------------------------

    base_dataset = VehicleClassificationDataset(
        root_dir=root_dir,
        transform=None,
    )

    print(
        f"\nDataset original: "
        f"{len(base_dataset):,} imágenes"
    )

    # --------------------------------------------------------
    # Balancear
    # --------------------------------------------------------

    balanced_samples = balance_samples(
        base_dataset.samples,
        seed=seed,
    )

    # --------------------------------------------------------
    # Split estratificado
    # --------------------------------------------------------

    (
        train_samples,
        val_samples,
        test_samples,
    ) = split_samples(
        balanced_samples,
        seed=seed,
    )

    # --------------------------------------------------------
    # Mostrar distribución
    # --------------------------------------------------------

    print_split_distribution(
        train_samples,
        val_samples,
        test_samples,
    )

    # --------------------------------------------------------
    # Crear transforms
    # --------------------------------------------------------

    train_transform = get_train_transforms(
        image_size=image_size,
        allow_upscale=allow_upscale,
    )

    val_transform = get_val_transforms(
        image_size=image_size,
        allow_upscale=allow_upscale,
    )

    test_transform = get_test_transforms(
        image_size=image_size,
        allow_upscale=allow_upscale,
    )

    # --------------------------------------------------------
    # Crear datasets
    # --------------------------------------------------------

    train_dataset = VehicleClassificationDataset(
        root_dir=root_dir,
        transform=train_transform,
        samples=train_samples,
    )

    val_dataset = VehicleClassificationDataset(
        root_dir=root_dir,
        transform=val_transform,
        samples=val_samples,
    )

    test_dataset = VehicleClassificationDataset(
        root_dir=root_dir,
        transform=test_transform,
        samples=test_samples,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# CREAR DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
):
    """
    Crea los DataLoader de Train, Validation y Test.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def get_dataloaders(
    root_dir,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    allow_upscale=ALLOW_UPSCALE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    seed=SEED,
):
    """
    Función principal para obtener directamente:

        train_loader
        val_loader
        test_loader
    """

    # --------------------------------------------------------
    # Crear datasets
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_datasets(
        root_dir=root_dir,
        image_size=image_size,
        allow_upscale=allow_upscale,
        seed=seed,
    )

    # --------------------------------------------------------
    # Crear DataLoaders
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )