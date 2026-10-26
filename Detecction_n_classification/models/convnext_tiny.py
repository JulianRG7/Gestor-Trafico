# models/convnext_tiny.py

import torch.nn as nn

from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


NUM_CLASSES = 10


def create_convnext_tiny(
    num_classes=NUM_CLASSES,
    pretrained=True,
):
    """
    Crea un ConvNeXt-Tiny para clasificación
    de vehículos.

    Parameters
    ----------
    num_classes : int
        Número de clases.

    pretrained : bool
        Si True, utiliza pesos preentrenados
        en ImageNet.
    """

    if pretrained:
        weights = ConvNeXt_Tiny_Weights.DEFAULT
    else:
        weights = None

    model = convnext_tiny(
        weights=weights
    )

    # --------------------------------------------------------
    # Reemplazar clasificador
    # --------------------------------------------------------

    in_features = model.classifier[2].in_features

    model.classifier[2] = nn.Linear(
        in_features,
        num_classes,
    )

    return model