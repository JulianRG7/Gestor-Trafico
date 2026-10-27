"""
yolo_train.py

Entrena un modelo YOLO (Ultralytics) sobre el dataset generado por
split_por_angulo.py (dataset_split/dataset.yaml: train=fuentes 2+3+4,
val/test=fuente 1 dividida por ángulo).

Al terminar el entrenamiento, evalúa el mejor checkpoint sobre el split
"test" (nunca visto durante el entrenamiento ni usado para early stopping)
y guarda un resumen de métricas por clase.

Requisitos:
    pip install ultralytics
"""

from pathlib import Path
import argparse
import json

import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN POR DEFECTO
# ============================================================

DATASET_YAML = Path("dataset_split/dataset.yaml")

# YOLO26 es la generación actual recomendada por Ultralytics (NMS-free,
# más rápida y precisa que YOLOv8/YOLO11 en igualdad de tamaño).
# Tamaños: n (nano) < s < m < l < x. "s" es un buen punto de partida
# para ~3000 imágenes; sube a "m"/"l" si tienes GPU grande y quieres
# exprimir precisión, o baja a "n" si necesitas iterar rápido.
MODEL = "yolo26n.pt"

EPOCHS = 50
CLOSE_MOSAIC = 20    # (default)
IMG_SIZE = 960       # las imágenes fuente son 1920x1080; 640 (default de
                      # Ultralytics) recorta detalle en vehículos lejanos
                      # o pequeños (ej. motos). 960-1280 suele ir mejor
                      # a cambio de más VRAM/tiempo por época.
BATCH = -1
PATIENCE = 30        # early stopping: épocas sin mejora antes de parar
PROJECT = "runs_traffic"
RUN_NAME = "yolo26_camera_angle_split-n3"
SEED = 7


# ============================================================
# DATA AUGMENTATION
# ============================================================
HSV_H = 0.015        # (default) el color no distingue tipo de vehículo, no hace falta tocarlo
HSV_S = 0.7          # (default)
HSV_V = 0.5          # un poco más que el default (0.4) si hay variación día/noche/clima
DEGREES = 0.0        # cámaras fijas -> casi no hay rotación real, déjalo en 0
TRANSLATE = 0.2      # (default)
SCALE = 0.5          # (default) importante: da variedad de tamaño (auto cercano vs lejano)
SHEAR = 0.0          # (default)
PERSPECTIVE = 0.0003   # (default) — si notas que el ángulo varía MUCHO entre fuentes, prueba 0.0003-0.0005
FLIPUD = 0.0         # (default) NUNCA actives esto: un auto de cabeza no existe en la realidad
FLIPLR = 0.5         # (default) volteo horizontal sí tiene sentido en vías
BGR = 0.0            # (default)
MOSAIC = 1.0         # (default)
MIXUP = 0.1          # subir desde 0 → regulariza un poco más con dataset chico
CUTMIX = 0.0         # (default) déjalo en 0 por ahora
COPY_PASTE = 0.1     # subir desde 0 → ayuda si tienes clases raras (motos, buses) para "duplicarlas" en contexto
COPY_PASTE_MODE = "flip"



CLS = 0.5            # peso de la loss de clasificación (default 0.5)
                      # subirlo empuja al modelo a ser más selectivo -> ayuda a precision

OPTIMIZER = "AdamW"   # 'auto' (default) o explícito:
                      # 'SGD','Adam','Adamax','AdamW','NAdam','RAdam','RMSProp'
LR0 = 0.001           # AdamW suele ir mejor con lr más chica que SGD (que usa 0.01 por defecto)

# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Entrena YOLO sobre dataset_split (split por ángulo de cámara)."
    )
    parser.add_argument("--data", type=Path, default=DATASET_YAML,
                         help="Ruta al dataset.yaml generado por split_por_angulo.py")
    parser.add_argument("--model", type=str, default=MODEL,
                         help="Checkpoint base, ej. yolo26n.pt, yolo26s.pt, yolo11s.pt ...")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--device", type=str, default=None,
                         help="'0', '0,1', 'cpu', 'mps'. Por defecto se autodetecta.")
    parser.add_argument("--project", type=str, default=PROJECT)
    parser.add_argument("--name", type=str, default=RUN_NAME)
    parser.add_argument("--resume", action="store_true",
                         help="Reanuda el último entrenamiento interrumpido en project/name.")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"No se encontró {args.data}. Corre primero split_por_angulo.py "
            f"para generar dataset_split/."
        )

    device = args.device or get_device()

    print("=" * 70)
    print("ENTRENAMIENTO YOLO")
    print("=" * 70)
    print(f"dataset  : {args.data}")
    print(f"modelo   : {args.model}")
    print(f"epochs   : {args.epochs}")
    print(f"imgsz    : {args.imgsz}")
    print(f"batch    : {args.batch}")
    print(f"patience : {args.patience}")
    print(f"device   : {device}")
    print()

    # --------------------------------------------------------
    # Reanudar entrenamiento interrumpido, si aplica
    # --------------------------------------------------------

    last_checkpoint = Path(args.project) / args.name / "weights" / "last.pt"

    if args.resume and last_checkpoint.exists():
        print(f"Reanudando desde: {last_checkpoint}")
        model = YOLO(str(last_checkpoint))
        train_kwargs = dict(resume=True)
    else:
        model = YOLO(args.model)
        train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        project=args.project,
        name=args.name,
        seed=SEED,
        plots=True,
        val=True,
        # --- augmentation ---
        hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
        degrees=DEGREES, translate=TRANSLATE, scale=SCALE,
        shear=SHEAR, perspective=PERSPECTIVE,
        flipud=FLIPUD, fliplr=FLIPLR, bgr=BGR,
        mosaic=MOSAIC, mixup=MIXUP, cutmix=CUTMIX,
        copy_paste=COPY_PASTE, copy_paste_mode=COPY_PASTE_MODE,
        close_mosaic=CLOSE_MOSAIC,
        cls=CLS,
        optimizer=OPTIMIZER,
        lr0=LR0,
    )

    results = model.train(**train_kwargs)

    # --------------------------------------------------------
    # Evaluación final sobre TEST
    # (held-out real: no se usó ni para entrenar ni para early stopping)
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUACIÓN FINAL SOBRE TEST")
    print("=" * 70)

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    if not best_weights.exists():
        print(f"AVISO: no se encontró {best_weights}, se usa el modelo en memoria.")
        best_model = model
    else:
        best_model = YOLO(str(best_weights))

    test_metrics = best_model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        device=device,
        plots=True,
    )

    print(f"\nmAP50    (test): {test_metrics.box.map50:.4f}")
    print(f"mAP50-95 (test): {test_metrics.box.map:.4f}")

    print("\nPor clase (mAP50-95):")
    per_class = {}
    for idx, name in test_metrics.names.items():
        ap = float(test_metrics.box.maps[idx]) if idx < len(test_metrics.box.maps) else float("nan")
        per_class[name] = ap
        print(f"  {name:<12} {ap:.4f}")

    # --------------------------------------------------------
    # Guardar resumen
    # --------------------------------------------------------

    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "best_weights": str(best_weights),
        "test_mAP50": float(test_metrics.box.map50),
        "test_mAP50_95": float(test_metrics.box.map),
        "test_per_class_mAP50_95": per_class,
    }

    summary_path = save_dir / "test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nPesos del mejor modelo : {best_weights}")
    print(f"Resumen de métricas    : {summary_path}")
    print(f"Gráficas y logs        : {save_dir}")


if __name__ == "__main__":
    main()