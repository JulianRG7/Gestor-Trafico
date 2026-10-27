import argparse
import inspect
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2, Bottleneck, C2f, C3
import ultralytics.nn.modules.block as block

def inspeccionar_yolo26(version="n", detailed=False):
    model_name = f"yolo26{version.lower()}.pt"

    print(f"--- Cargando arquitectura para: {model_name} ---")

    model = YOLO(model_name)

    print("\n=== Resumen general ===")
    model.info(detailed=detailed)

    print("\n=== Estructura PyTorch (Sequential) ===")
    print(model.model)

    print("\n=== Código de C3k2 ===")
    print(inspect.getsource(C3k2))

    print("\n=== Código de C2f ===")
    print(inspect.getsource(C2f))

    print(inspect.getsource(Bottleneck))

    print("\n=== Clases relacionadas ===")
    print([nombre for nombre in dir(block) if "C3" in nombre])

    print("\n=== Código de C3k ===")
    print(inspect.getsource(block.C3k))

    print("\n=== Código de C3 ===")
    print(inspect.getsource(block.C3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspeccionar la arquitectura de YOLO26"
    )

    parser.add_argument(
        "-s", "--size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="Escala del modelo"
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Muestra información detallada"
    )

    args = parser.parse_args()

    inspeccionar_yolo26(
        version=args.size,
        detailed=args.detailed
    )