import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import confusion_matrix
from training.metrics import calculate_metrics


def evaluate_model(
    model,
    dataloader,
    device,
    num_classes
):
    """
    Evalúa un modelo sobre un DataLoader.

    Retorna:
        metrics
        all_targets
        all_predictions
    """

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = torch.argmax(outputs, dim=1)

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    average_loss = total_loss / total_samples

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
        num_classes
    )

    # Agregamos la pérdida a las métricas
    metrics["loss"] = float(average_loss)

    return metrics, all_targets, all_predictions


def save_confusion_matrix(
    metrics,
    class_names,
    save_path,
    title
):
    """
    Guarda una matriz de confusión normalizada por clase.

    Cada fila representa una clase real y suma 100%.

    Las celdas muestran porcentajes en lugar
    de cantidades absolutas.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # ========================================================
    # OBTENER MATRIZ DE CONFUSIÓN
    # ========================================================

    cm = np.array(
        metrics["confusion_matrix"],
        dtype=float
    )

    # ========================================================
    # NORMALIZACIÓN POR FILA
    # ========================================================

    row_sums = cm.sum(axis=1, keepdims=True)

    # Evitar división entre cero
    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm),
        where=row_sums != 0
    )

    # Convertir a porcentaje
    cm_percentage = cm_normalized * 100.0

    # ========================================================
    # CREAR FIGURA
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    # ========================================================
    # MATRIZ EN ESCALA BLANCO → AZUL
    # ========================================================

    image = ax.imshow(
        cm_percentage,
        cmap="Blues",
        vmin=0,
        vmax=100
    )

    # ========================================================
    # COLORBAR
    # ========================================================

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Percentage (%)"
    )

    # ========================================================
    # ETIQUETAS
    # ========================================================

    ax.set_xlabel(
        "Predicted Class",
        fontsize=12
    )

    ax.set_ylabel(
        "True Class",
        fontsize=12
    )

    ax.set_title(
        title,
        fontsize=14,
        pad=15
    )

    # ========================================================
    # NOMBRES DE CLASES
    # ========================================================

    ax.set_xticks(
        np.arange(len(class_names))
    )

    ax.set_yticks(
        np.arange(len(class_names))
    )

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        class_names
    )

    # ========================================================
    # MOSTRAR PORCENTAJES EN LAS CELDAS
    # ========================================================

    for i in range(cm_percentage.shape[0]):

        for j in range(cm_percentage.shape[1]):

            percentage = cm_percentage[i, j]

            # Elegir automáticamente el color del texto
            # dependiendo de la intensidad del fondo.
            if percentage > 50:
                text_color = "white"
            else:
                text_color = "black"

            ax.text(
                j,
                i,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10
            )

    # ========================================================
    # AJUSTES
    # ========================================================

    ax.set_xlim(
        -0.5,
        len(class_names) - 0.5
    )

    ax.set_ylim(
        len(class_names) - 0.5,
        -0.5
    )

    plt.tight_layout()

    # ========================================================
    # GUARDAR
    # ========================================================

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def save_loss_curve(
    history,
    save_path,
    model_name
):
    """
    Guarda la curva de pérdida de entrenamiento
    y validación.

    Se generan dos curvas:

        Train Loss
        Validation Loss
    """

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    epochs = range(
        1,
        len(train_loss) + 1
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_loss,
        label="Train Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"Loss Curve - {model_name}"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def save_metrics(
    metrics,
    save_path
):
    """
    Guarda las métricas en formato JSON.
    """

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )