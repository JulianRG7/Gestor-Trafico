# scripts/test_models.py

import os
import sys
from pathlib import Path

import torch


# ============================================================
# CONFIGURAR RUTA DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

from datasets.dataloader import get_dataloaders

from models.efficientnet_b1 import (
    create_efficientnet_b1,
)

from models.resnet50 import (
    create_resnet50,
)

from models.convnext_tiny import (
    create_convnext_tiny,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

print(f"Ruta del proyecto: {PROJECT_ROOT}")

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "VehicleClassificationDataset"
)

BATCH_SIZE = 4

NUM_CLASSES = 10

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================

MODEL_CONFIGS = {

    "EfficientNet-B1": {
        "image_size": 240,
        "create_model": create_efficientnet_b1,
    },

    "ResNet-50": {
        "image_size": 224,
        "create_model": create_resnet50,
    },

    "ConvNeXt-Tiny": {
        "image_size": 224,
        "create_model": create_convnext_tiny,
    },
}


# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

def count_parameters(model):
    """
    Cuenta el número total de parámetros.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_trainable_parameters(model):
    """
    Cuenta únicamente los parámetros entrenables.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# TEST DE UN MODELO
# ============================================================

def test_model(
    model,
    model_name,
    images,
    expected_image_size,
):
    """
    Comprueba que un modelo:

        1. Se puede mover al dispositivo.
        2. Recibe la dimensión esperada.
        3. Realiza correctamente el forward pass.
        4. Produce 10 salidas por imagen.
        5. No produce NaN ni Inf.
    """

    print("\n" + "=" * 70)
    print(f"TEST — {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    model = model.to(DEVICE)

    model.eval()

    # --------------------------------------------------------
    # Parámetros
    # --------------------------------------------------------

    total_params = count_parameters(model)

    trainable_params = count_trainable_parameters(
        model
    )

    print(
        f"Parámetros totales:      "
        f"{total_params:,}"
    )

    print(
        f"Parámetros entrenables:  "
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    images = images.to(DEVICE)

    print(
        f"\nInput shape:  "
        f"{tuple(images.shape)}"
    )

    # --------------------------------------------------------
    # Verificar tamaño de imagen
    # --------------------------------------------------------

    actual_height = images.shape[2]
    actual_width = images.shape[3]

    expected_shape = (
        BATCH_SIZE,
        3,
        expected_image_size,
        expected_image_size,
    )

    assert images.shape == expected_shape, (

        "\nDimensión de entrada incorrecta.\n"

        f"Esperado: {expected_shape}\n"

        f"Obtenido: {tuple(images.shape)}"
    )

    print(
        f"✓ Input correcto: "
        f"{expected_image_size} × "
        f"{expected_image_size}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(images)

    print(
        f"Output shape: "
        f"{tuple(outputs.shape)}"
    )

    # --------------------------------------------------------
    # Verificar output
    # --------------------------------------------------------

    expected_output_shape = (
        images.shape[0],
        NUM_CLASSES,
    )

    assert outputs.shape == expected_output_shape, (

        "\nOutput incorrecto.\n"

        f"Esperado: {expected_output_shape}\n"

        f"Obtenido: {tuple(outputs.shape)}"
    )

    print(
        f"✓ Output correcto: "
        f"[batch_size, {NUM_CLASSES}]"
    )

    # --------------------------------------------------------
    # NaN / Inf
    # --------------------------------------------------------

    assert not torch.isnan(outputs).any(), (
        "El modelo produjo NaN."
    )

    assert not torch.isinf(outputs).any(), (
        "El modelo produjo Inf."
    )

    print("✓ No hay NaN ni Inf")

    # --------------------------------------------------------
    # Dtype
    # --------------------------------------------------------

    assert outputs.dtype == torch.float32, (

        f"Dtype inesperado: "
        f"{outputs.dtype}"
    )

    print("✓ Output dtype: float32")

    return model


# ============================================================
# CREAR DATALOADER PARA UN MODELO
# ============================================================

def create_test_loader(image_size):
    """
    Crea un DataLoader utilizando el tamaño de entrada
    correspondiente al modelo.
    """

    train_loader, _, _ = get_dataloaders(

        root_dir=DATASET_ROOT,

        batch_size=BATCH_SIZE,

        image_size=image_size,

        seed=SEED,
    )

    return train_loader


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("              TEST DE MODELOS")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "CUDA no disponible. "
            "Se utilizará CPU."
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VERIFICANDO DATASET")
    print("=" * 70)

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"\nNo existe el dataset:\n"
            f"{DATASET_ROOT}"
        )

    print(
        f"✓ Dataset encontrado:\n"
        f"{DATASET_ROOT}"
    )

    # ========================================================
    # TEST DE CADA MODELO
    # ========================================================

    for model_name, config in MODEL_CONFIGS.items():

        image_size = config["image_size"]

        create_model = config["create_model"]

        print("\n")
        print("-" * 70)

        print(
            f"Preparando {model_name}"
        )

        print(
            f"Tamaño de entrada: "
            f"{image_size} × {image_size}"
        )

        # ----------------------------------------------------
        # DataLoader específico
        # ----------------------------------------------------

        train_loader = create_test_loader(
            image_size=image_size
        )

        images, labels = next(
            iter(train_loader)
        )

        print(
            f"Batch obtenido: "
            f"{tuple(images.shape)}"
        )

        print(
            f"Labels: "
            f"{tuple(labels.shape)}"
        )

        # ----------------------------------------------------
        # Crear modelo
        # ----------------------------------------------------

        model = create_model(
            num_classes=NUM_CLASSES,
            pretrained=True,
        )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        test_model(
            model=model,
            model_name=model_name,
            images=images,
            expected_image_size=image_size,
        )

        # ----------------------------------------------------
        # Liberar memoria
        # ----------------------------------------------------

        del model
        del train_loader
        del images
        del labels

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ========================================================
    # RESULTADO
    # ========================================================

    print("\n")
    print("=" * 70)
    print("             TODOS LOS MODELOS PASARON")
    print("=" * 70)

    print(
        "\nEfficientNet-B1  ✓  240 × 240"
    )

    print(
        "ResNet-50        ✓  224 × 224"
    )

    print(
        "ConvNeXt-Tiny    ✓  224 × 224"
    )

    print(
        "\nLos modelos están listos "
        "para la etapa de entrenamiento.\n"
    )