"""
split_balanceado_roi_90.py

Divide el dataset YOLO aplicando enmascarado de ROI.
Respeta los bounding boxes si al menos el X% (ej. 90%) de su área
cae dentro del polígono irregular definido para esa cámara.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import random
from shapely.geometry import Polygon

# ============================================================
# CONFIGURACIÓN
# ============================================================

CLASS_NAMES = {
    0: "Car",
    1: "Motorcycle",
    2: "Passenger",
    3: "Cargo",
    4: "Ambulance",
    5: "Police",
    6: "FireTruck",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ALL_SOURCES = [
    Path("YOLODataset-Camera_Free3-L-V1"),
    Path("YOLODataset-Camera_Free3-L-V2"),
    Path("YOLODataset"),
    Path("YOLODataset-Camera_Free-V1"),
    Path("YOLODataset-Camera_Free2"),
    Path("YOLODataset-Camera_Free-11")
]

OUTPUT_DIR = Path("./dataset_split")
ROIS_JSON_PATH = "rois.json"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.50
SEED = 42
MAX_IMAGES_PER_SOURCE = "all"

# NUEVO: Porcentaje mínimo de área que debe estar dentro del ROI (0.90 = 90%)
MIN_BBOX_INTERSECTION = 0.90

# Cargar ROIs al iniciar
try:
    with open(ROIS_JSON_PATH, "r") as f:
        ROIS = json.load(f)
    print(f"ROIs cargados correctamente desde {ROIS_JSON_PATH}")
except FileNotFoundError:
    print(f"ADVERTENCIA: No se encontró {ROIS_JSON_PATH}. No se aplicarán máscaras.")
    ROIS = {}

# ============================================================
# FUNCIONES DE LECTURA, LIMPIEZA Y ROI
# ============================================================

def get_pairs(source):
    images_dir = source / "images"
    labels_dir = source / "labels"

    if not images_dir.exists():
        print(f"AVISO: no existe {images_dir}, se omite esta fuente.")
        return []

    pairs = []
    for img_path in sorted(images_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        pairs.append((img_path, label_path))

    return pairs

def is_box_in_roi(x, y, w, h, roi_points, threshold=MIN_BBOX_INTERSECTION):
    """
    Calcula el área de intersección entre el Bounding Box y el ROI.
    Retorna True si el porcentaje de intersección es mayor o igual al threshold.
    """
    if not roi_points:
        return True 

    # Crear el polígono del ROI usando Shapely
    roi_poly = Polygon(roi_points)
    
    # Prevenir errores si el usuario dibujó un polígono que se cruza a sí mismo
    if not roi_poly.is_valid:
        roi_poly = roi_poly.buffer(0)

    # Calcular las 4 esquinas del bounding box (coordenadas normalizadas)
    xmin, xmax = x - (w / 2), x + (w / 2)
    ymin, ymax = y - (h / 2), y + (h / 2)
    
    # Crear el polígono del Bounding Box
    bbox_poly = Polygon([
        (xmin, ymin), 
        (xmax, ymin), 
        (xmax, ymax), 
        (xmin, ymax)
    ])

    bbox_area = bbox_poly.area
    if bbox_area == 0:
        return False

    # Calcular qué tanta área del bbox está dentro del ROI
    intersection_area = bbox_poly.intersection(roi_poly).area
    ratio = intersection_area / bbox_area

    return ratio >= threshold

def clean_label_lines(label_path, roi_points):
    if not label_path.exists():
        return [], []

    kept, dropped = [], []
    with open(label_path, "r") as f:
        lines = f.readlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        values = line.split()
        if len(values) != 5:
            dropped.append(line)
            continue
        try:
            class_id = int(values[0])
            x, y, w, h = (float(v) for v in values[1:])
        except ValueError:
            dropped.append(line)
            continue

        valid = (
            class_id in CLASS_NAMES
            and 0 <= x <= 1
            and 0 <= y <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        )

        # Validación de Área de Intersección
        if valid and not is_box_in_roi(x, y, w, h, roi_points):
            valid = False

        if valid:
            kept.append((line, class_id))
        else:
            dropped.append(line)

    return kept, dropped

# ============================================================
# ESTRATIFICACIÓN MULTI-ETIQUETA
# ============================================================

def iterative_stratified_split(images_classes, ratio=0.5, seed=SEED):
    rng = random.Random(seed)
    images = list(images_classes.keys())
    rng.shuffle(images)

    all_classes = set()
    for c in images_classes.values():
        all_classes |= c

    class_totals = defaultdict(int)
    for c in images_classes.values():
        for cid in c:
            class_totals[cid] += 1

    target_a = {cid: class_totals[cid] * ratio for cid in all_classes}
    target_b = {cid: class_totals[cid] * (1 - ratio) for cid in all_classes}

    current_a = defaultdict(int)
    current_b = defaultdict(int)
    set_a, set_b = set(), set()

    no_class = [img for img in images if not images_classes[img]]
    remaining = set(images) - set(no_class)

    while remaining:
        class_remaining_count = defaultdict(int)
        for img in remaining:
            for cid in images_classes[img]:
                class_remaining_count[cid] += 1

        if not class_remaining_count:
            break

        min_class = min(class_remaining_count, key=lambda c: class_remaining_count[c])
        candidates = [img for img in remaining if min_class in images_classes[img]]
        rng.shuffle(candidates)

        for img in candidates:
            deficit_a = target_a[min_class] - current_a[min_class]
            deficit_b = target_b[min_class] - current_b[min_class]

            if deficit_a > deficit_b:
                chosen = "A"
            elif deficit_b > deficit_a:
                chosen = "B"
            else:
                chosen = "A" if len(set_a) <= len(set_b) else "B"

            if chosen == "A":
                set_a.add(img)
                for cid in images_classes[img]:
                    current_a[cid] += 1
            else:
                set_b.add(img)
                for cid in images_classes[img]:
                    current_b[cid] += 1

            remaining.discard(img)

    rng.shuffle(no_class)
    for img in no_class:
        if len(set_a) <= len(set_b):
            set_a.add(img)
        else:
            set_b.add(img)

    return set_a, set_b

# ============================================================
# ESCRITURA DE ARCHIVOS (CON OPENCV Y ROI)
# ============================================================

def place_image(src, dst, roi_points):
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    img = cv2.imread(str(src))
    if img is None:
        return
        
    if roi_points:
        h, w = img.shape[:2]
        
        pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in roi_points], np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        img = cv2.bitwise_and(img, img, mask=mask)
        
    cv2.imwrite(str(dst), img)

def place_label(label_path, dst, drop_counter, roi_points):
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept, dropped = clean_label_lines(label_path, roi_points)
    drop_counter[0] += len(dropped)
    with open(dst, "w") as f:
        for line, _ in kept:
            f.write(line + "\n")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print(f"SPLIT BALANCEADO - ROI TOLERANCIA BBOX ({MIN_BBOX_INTERSECTION*100}%)")
    print("=" * 70)

    images_classes = {}
    lookup = {}
    total_images = 0

    for source in ALL_SOURCES:
        pairs = get_pairs(source)
        
        if str(MAX_IMAGES_PER_SOURCE).lower() != "all":
            try:
                limit = int(MAX_IMAGES_PER_SOURCE)
                pairs = pairs[:limit]
            except ValueError:
                print("Error: MAX_IMAGES_PER_SOURCE debe ser un número entero o 'all'.")

        source_name = source.name
        roi_points = ROIS.get(source_name, [])
        
        print(f"Procesando {source}: {len(pairs)} imágenes | ROI: {'Sí' if roi_points else 'No'}")
        total_images += len(pairs)
        
        for img_path, label_path in pairs:
            uid = f"{source_name}__{img_path.stem}"
            kept, _ = clean_label_lines(label_path, roi_points)
            images_classes[uid] = {class_id for _, class_id in kept}
            lookup[uid] = (img_path, label_path, source_name)

    print(f"\nTotal de imágenes combinadas: {total_images}")

    if total_images == 0:
        print("No se encontraron imágenes para procesar. Verifica las rutas.")
        return

    train_ids, temp_ids = iterative_stratified_split(images_classes, ratio=TRAIN_RATIO, seed=SEED)
    
    temp_classes = {uid: images_classes[uid] for uid in temp_ids}
    val_ids, test_ids = iterative_stratified_split(temp_classes, ratio=VAL_RATIO, seed=SEED)

    print(f"  -> train: {len(train_ids)} imágenes")
    print(f"  -> val:   {len(val_ids)} imágenes")
    print(f"  -> test:  {len(test_ids)} imágenes")

    print(f"\nEscribiendo imágenes procesadas en {OUTPUT_DIR.resolve()} ...")
    drop_counter = [0]

    def dump(uid, split):
        img_path, label_path, source_name = lookup[uid]
        roi_points = ROIS.get(source_name, [])
        
        dst_img = OUTPUT_DIR / split / "images" / f"{uid}{img_path.suffix.lower()}"
        dst_lbl = OUTPUT_DIR / split / "labels" / f"{uid}.txt"
        
        place_image(img_path, dst_img, roi_points)
        place_label(label_path, dst_lbl, drop_counter, roi_points)

    for uid in train_ids:
        dump(uid, "train")
    for uid in val_ids:
        dump(uid, "val")
    for uid in test_ids:
        dump(uid, "test")

    print(f"Líneas descartadas por estar debajo del {MIN_BBOX_INTERSECTION*100}% dentro del ROI: {drop_counter[0]}")

    yaml_path = OUTPUT_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR.resolve()}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write("names:\n")
        for cid, name in CLASS_NAMES.items():
            f.write(f"  {cid}: {name}\n")

    print(f"dataset.yaml actualizado en: {yaml_path.resolve()}")
    print("\n¡Listo! El split ha finalizado con la tolerancia de área requerida.")

if __name__ == "__main__":
    main()