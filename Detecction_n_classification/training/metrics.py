# training/metrics.py

import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# MÉTRICAS DE CLASIFICACIÓN
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    num_classes,
):
    """
    Calcula las métricas de clasificación.

    Parameters
    ----------
    y_true : array-like
        Labels reales.

    y_pred : array-like
        Predicciones del modelo.

    num_classes : int
        Número de clases.

    Returns
    -------
    dict
        Diccionario con métricas globales y por clase.
    """

    labels = np.arange(num_classes)

    # --------------------------------------------------------
    # Métricas globales
    # --------------------------------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Métricas por clase
    # --------------------------------------------------------

    precision_per_class = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    recall_per_class = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    f1_per_class = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Matriz de confusión
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),

        "precision_per_class": (
            precision_per_class.tolist()
        ),

        "recall_per_class": (
            recall_per_class.tolist()
        ),

        "f1_per_class": (
            f1_per_class.tolist()
        ),

        "confusion_matrix": cm.tolist(),
    }