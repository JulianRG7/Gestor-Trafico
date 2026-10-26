# models/resnet50.py

import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)


NUM_CLASSES = 10


def create_resnet50(
    num_classes=NUM_CLASSES,
    pretrained=True,
):
    """
    Crea un ResNet-50 para clasificación
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
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = resnet50(
        weights=weights
    )

    # --------------------------------------------------------
    # Reemplazar capa totalmente conectada
    # --------------------------------------------------------

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes,
    )

    return model