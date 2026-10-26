from pathlib import Path
import sys
import json

import torch
from torch.utils.data import DataLoader

# ============================================================
# AGREGAR ROOT DEL PROYECTO AL PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORTS DEL PROYECTO
# ============================================================

from datasets.dataloader import get_dataloaders
from datasets.vehicle_dataset import (
    VehicleClassificationDataset,
    CLASS_NAMES,
)

from datasets.transforms import get_val_transforms

from models.efficientnet_b1 import create_efficientnet_b1
from models.resnet50 import create_resnet50
from models.convnext_tiny import create_convnext_tiny

from training.trainer import train_model

from training.evaluator import (
    evaluate_model,
    save_confusion_matrix,
    save_metrics,
    save_loss_curve,
)


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "VehicleClassificationDataset"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

METRICS_DIR = (
    REPORTS_DIR
    / "metrics"
)

CONFUSION_DIR = (
    REPORTS_DIR
    / "confusion_matrices"
)

CURVES_DIR = (
    REPORTS_DIR
    / "curves"
)


# ============================================================
# HIPERPARÁMETROS
# ============================================================

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

SEED = 42

NUM_CLASSES = 10

NUM_WORKERS = 4

PIN_MEMORY = True


# ============================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================

MODEL_CONFIGS = {

    "efficientnet_b1": {
        "image_size": 240,
        "create_model": create_efficientnet_b1,
    },

    "resnet50": {
        "image_size": 224,
        "create_model": create_resnet50,
    },

    "convnext_tiny": {
        "image_size": 224,
        "create_model": create_convnext_tiny,
    },
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CREAR DIRECTORIOS
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONFUSION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CURVES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FUNCIÓN PARA EVALUAR TRAIN SIN AUGMENTATION
# ============================================================

def create_train_eval_loader(
    train_loader,
    image_size
):
    """
    Crea un DataLoader para evaluar el conjunto de entrenamiento
    utilizando transformaciones deterministas.

    Esto evita que las métricas finales de Train se calculen
    sobre imágenes modificadas aleatoriamente mediante:

        RandomHorizontalFlip
        RandomRotation
        ColorJitter
    """

    train_dataset = train_loader.dataset

    # Recuperamos las muestras originales del dataset
    train_samples = train_dataset.samples

    # Dataset nuevo con transformaciones deterministas
    train_eval_dataset = VehicleClassificationDataset(
        root_dir=DATASET_ROOT,
        transform=get_val_transforms(
            image_size=image_size,
            allow_upscale=True
        ),
        samples=train_samples
    )

    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_eval_loader


# ============================================================
# FUNCIÓN PRINCIPAL DE EXPERIMENTO
# ============================================================

def run_experiment(
    model_name,
    model_config
):
    """
    Ejecuta el experimento completo para un modelo:

        1. Crear DataLoaders
        2. Crear modelo
        3. Entrenar
        4. Guardar curva de pérdida
        5. Cargar mejor checkpoint
        6. Evaluar Train
        7. Evaluar Validation
        8. Evaluar Test
        9. Guardar métricas
        10. Guardar matrices de confusión
    """

    print()
    print("=" * 70)
    print(f"MODELO: {model_name}")
    print("=" * 70)

    image_size = model_config["image_size"]

    create_model = model_config["create_model"]

    print(f"Image size: {image_size}x{image_size}")
    print(f"Device: {DEVICE}")
    print()


    # ========================================================
    # CREAR DATALOADERS
    # ========================================================

    print("Creando DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        image_size=image_size,
        allow_upscale=True,
        seed=SEED,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print("DataLoaders creados.")


    # ========================================================
    # TRAIN EVAL LOADER
    # ========================================================

    print(
        "Creando DataLoader de evaluación para Train..."
    )

    train_eval_loader = create_train_eval_loader(
        train_loader=train_loader,
        image_size=image_size
    )

    print(
        "Train evaluation DataLoader creado."
    )


    # ========================================================
    # CREAR MODELO
    # ========================================================

    print()
    print("Creando modelo...")

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=True
    )

    model = model.to(DEVICE)

    print(
        f"Modelo {model_name} creado."
    )


    # ========================================================
    # ENTRENAMIENTO
    # ========================================================

    print()
    print("-" * 70)
    print("INICIANDO ENTRENAMIENTO")
    print("-" * 70)

    model, history, best_checkpoint = train_model(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        device=DEVICE,

        epochs=EPOCHS,

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        scheduler=None,

        checkpoint_dir=CHECKPOINT_DIR,

        model_name=model_name,

        num_classes=NUM_CLASSES
    )


    # ========================================================
    # GUARDAR CURVA DE PÉRDIDA
    # ========================================================

    loss_curve_path = (
        CURVES_DIR
        / f"{model_name}_loss.png"
    )

    save_loss_curve(
        history=history,
        save_path=loss_curve_path,
        model_name=model_name
    )

    print()
    print(
        f"Curva de pérdida guardada en:\n"
        f"{loss_curve_path}"
    )


    # ========================================================
    # CARGAR MEJOR CHECKPOINT
    # ========================================================

    print()
    print("Cargando mejor checkpoint...")

    checkpoint = torch.load(
        best_checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Mejor checkpoint: "
        f"epoch {checkpoint['epoch']}"
    )

    print(
        f"Mejor Validation Macro-F1: "
        f"{checkpoint['best_macro_f1']:.4f}"
    )


    # ========================================================
    # EVALUACIÓN FINAL
    # ========================================================

    print()
    print("-" * 70)
    print("EVALUACIÓN FINAL")
    print("-" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("Evaluando TRAIN...")

    train_metrics, train_targets, train_predictions = (
        evaluate_model(
            model=model,
            dataloader=train_eval_loader,
            device=DEVICE,
            num_classes=NUM_CLASSES
        )
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("Evaluando VALIDATION...")

    val_metrics, val_targets, val_predictions = (
        evaluate_model(
            model=model,
            dataloader=val_loader,
            device=DEVICE,
            num_classes=NUM_CLASSES
        )
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("Evaluando TEST...")

    test_metrics, test_targets, test_predictions = (
        evaluate_model(
            model=model,
            dataloader=test_loader,
            device=DEVICE,
            num_classes=NUM_CLASSES
        )
    )


    # ========================================================
    # MOSTRAR MÉTRICAS
    # ========================================================

    print()
    print("=" * 70)
    print(f"RESULTADOS: {model_name}")
    print("=" * 70)

    print()
    print("TRAIN")
    print(
        f"Loss:       {train_metrics['loss']:.4f}"
    )
    print(
        f"Precision:  {train_metrics['precision']:.4f}"
    )
    print(
        f"Recall:     {train_metrics['recall']:.4f}"
    )
    print(
        f"F1:         {train_metrics['f1']:.4f}"
    )
    print(
        f"Macro-F1:   {train_metrics['macro_f1']:.4f}"
    )

    print()
    print("VALIDATION")
    print(
        f"Loss:       {val_metrics['loss']:.4f}"
    )
    print(
        f"Precision:  {val_metrics['precision']:.4f}"
    )
    print(
        f"Recall:     {val_metrics['recall']:.4f}"
    )
    print(
        f"F1:         {val_metrics['f1']:.4f}"
    )
    print(
        f"Macro-F1:   {val_metrics['macro_f1']:.4f}"
    )

    print()
    print("TEST")
    print(
        f"Loss:       {test_metrics['loss']:.4f}"
    )
    print(
        f"Precision:  {test_metrics['precision']:.4f}"
    )
    print(
        f"Recall:     {test_metrics['recall']:.4f}"
    )
    print(
        f"F1:         {test_metrics['f1']:.4f}"
    )
    print(
        f"Macro-F1:   {test_metrics['macro_f1']:.4f}"
    )


    # ========================================================
    # GUARDAR MÉTRICAS
    # ========================================================

    metrics_data = {

        "model": model_name,

        "image_size": image_size,

        "best_epoch": checkpoint["epoch"],

        "best_validation_macro_f1":
            checkpoint["best_macro_f1"],

        "train": train_metrics,

        "validation": val_metrics,

        "test": test_metrics
    }

    metrics_path = (
        METRICS_DIR
        / f"{model_name}.json"
    )

    save_metrics(
        metrics=metrics_data,
        save_path=metrics_path
    )

    print()
    print(
        f"Métricas guardadas en:\n"
        f"{metrics_path}"
    )


    # ========================================================
    # MATRICES DE CONFUSIÓN
    # ========================================================

    print()
    print("Guardando matrices de confusión...")


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_cm_path = (
        CONFUSION_DIR
        / f"{model_name}_train.png"
    )

    save_confusion_matrix(
        metrics=train_metrics,
        class_names=CLASS_NAMES,
        save_path=train_cm_path,
        title=f"{model_name} - Train Confusion Matrix"
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_cm_path = (
        CONFUSION_DIR
        / f"{model_name}_val.png"
    )

    save_confusion_matrix(
        metrics=val_metrics,
        class_names=CLASS_NAMES,
        save_path=val_cm_path,
        title=f"{model_name} - Validation Confusion Matrix"
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_cm_path = (
        CONFUSION_DIR
        / f"{model_name}_test.png"
    )

    save_confusion_matrix(
        metrics=test_metrics,
        class_names=CLASS_NAMES,
        save_path=test_cm_path,
        title=f"{model_name} - Test Confusion Matrix"
    )


    print(
        "Matrices de confusión guardadas."
    )


    # ========================================================
    # GUARDAR HISTORY
    # ========================================================

    history_path = (
        METRICS_DIR
        / f"{model_name}_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False
        )


    print(
        f"Historial guardado en:\n"
        f"{history_path}"
    )


    # ========================================================
    # RETORNAR RESULTADOS
    # ========================================================

    return {

        "model": model_name,

        "image_size": image_size,

        "best_epoch":
            checkpoint["epoch"],

        "best_validation_macro_f1":
            checkpoint["best_macro_f1"],

        "train": train_metrics,

        "validation": val_metrics,

        "test": test_metrics
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VEHICLE CLASSIFICATION - MODEL COMPARISON")
    print("=" * 70)

    print()
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Device:  {DEVICE}")
    print(f"Epochs:  {EPOCHS}")
    print(f"Batch:   {BATCH_SIZE}")
    print(f"Seed:    {SEED}")

    print()
    print("Modelos a entrenar:")

    for model_name, config in MODEL_CONFIGS.items():

        print(
            f"  - {model_name}: "
            f"{config['image_size']}x{config['image_size']}"
        )


    # ========================================================
    # EJECUTAR EXPERIMENTOS
    # ========================================================

    all_results = {}


    for model_name, model_config in MODEL_CONFIGS.items():

        results = run_experiment(
            model_name=model_name,
            model_config=model_config
        )

        all_results[model_name] = results


    # ========================================================
    # GUARDAR COMPARACIÓN
    # ========================================================

    comparison_path = (
        METRICS_DIR
        / "comparison.json"
    )

    with open(
        comparison_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False
        )


    # ========================================================
    # RESUMEN FINAL
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("COMPARACIÓN FINAL")
    print("=" * 70)

    print()

    for model_name, results in all_results.items():

        test_metrics = results["test"]

        print(
            f"{model_name:20s} | "
            f"Precision: {test_metrics['precision']:.4f} | "
            f"Recall: {test_metrics['recall']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | "
            f"Macro-F1: {test_metrics['macro_f1']:.4f}"
        )

    print()
    print(
        f"Comparación guardada en:\n"
        f"{comparison_path}"
    )

    print()
    print("=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()