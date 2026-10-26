from pathlib import Path

import torch

from training.evaluator import evaluate_model


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    """
    Entrena el modelo durante una época.
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    epoch_loss = (
        total_loss / total_samples
    )

    epoch_accuracy = (
        total_correct / total_samples
    )

    return epoch_loss, epoch_accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    learning_rate,
    weight_decay,
    scheduler=None,
    checkpoint_dir="checkpoints",
    model_name="model",
    num_classes=10
):
    """
    Entrena un modelo y guarda el mejor checkpoint
    según Macro-F1 de validación.
    """

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Scheduler por defecto
    if scheduler is None:

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )

    history = {
        "train_loss": [],
        "train_accuracy": [],

        "val_loss": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_macro_f1": [],

        "lr": []
    }

    best_macro_f1 = -1.0

    checkpoint_dir = Path(
        checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_checkpoint = (
        checkpoint_dir
        / f"{model_name}_best.pth"
    )

    for epoch in range(
        1,
        epochs + 1
    ):

        # ==========================================
        # TRAIN
        # ==========================================

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # ==========================================
        # VALIDATION
        # ==========================================

        val_metrics, _, _ = evaluate_model(
            model,
            val_loader,
            device,
            num_classes
        )

        # ==========================================
        # SCHEDULER
        # ==========================================

        scheduler.step(
            val_metrics["loss"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # ==========================================
        # GUARDAR HISTORIAL
        # ==========================================

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_metrics["loss"]
        )

        history["val_precision"].append(
            val_metrics["precision"]
        )

        history["val_recall"].append(
            val_metrics["recall"]
        )

        history["val_f1"].append(
            val_metrics["f1"]
        )

        history["val_macro_f1"].append(
            val_metrics["macro_f1"]
        )

        history["lr"].append(
            current_lr
        )

        # ==========================================
        # MOSTRAR RESULTADOS
        # ==========================================

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # ==========================================
        # GUARDAR MEJOR MODELO
        # ==========================================

        if val_metrics["macro_f1"] > best_macro_f1:

            best_macro_f1 = (
                val_metrics["macro_f1"]
            )

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "best_macro_f1":
                        best_macro_f1,

                    "train_loss":
                        train_loss,

                    "val_metrics":
                        val_metrics
                },
                best_checkpoint
            )

            print(
                f"  ✓ Nuevo mejor modelo "
                f"(Macro-F1: {best_macro_f1:.4f})"
            )

    return (
        model,
        history,
        best_checkpoint
    )