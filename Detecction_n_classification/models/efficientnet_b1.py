# models/efficientnet_b1.py

import torch.nn as nn
from torchvision.models import (
    efficientnet_b1,
    EfficientNet_B1_Weights,
)


NUM_CLASSES = 10


def create_efficientnet_b1(
    num_classes=NUM_CLASSES,
    pretrained=True,
):
    """
    Crea un EfficientNet-B1 para clasificación
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
        weights = EfficientNet_B1_Weights.DEFAULT
    else:
        weights = None

    model = efficientnet_b1(
        weights=weights
    )

    # --------------------------------------------------------
    # Reemplazar clasificador
    # --------------------------------------------------------

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model