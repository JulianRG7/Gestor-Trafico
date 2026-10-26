import os

# IMPORTANTE: estas variables deben fijarse ANTES de importar numpy,
# scipy, sklearn o torch, porque esas librerías leen el número de
# threads a usar en el momento del import.
#
# Se fuerza a 1 thread (no un valor "óptimo" mayor) porque el
# OpenBLAS empaquetado en el wheel de numpy (confirmado con
# numpy.show_config(): MAX_THREADS=2, NO_AFFINITY=1) tiene un bug
# conocido: si el proceso ya inicializó un contexto CUDA (como aquí,
# al cargar el modelo en GPU), la primera vez que OpenBLAS intenta
# crear su pool de threads internamente se puede quedar bloqueada
# de forma indefinida (deadlock real, no solo lentitud). Con 1
# thread, OpenBLAS nunca crea ese pool y el problema desaparece.
# El costo en rendimiento es insignificante para el tamaño de estos
# cálculos (PCA a 3 componentes, etc.).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from pathlib import Path

import numpy as np
import pandas as pd

# IMPORTANTE: forzar backend no interactivo ANTES de importar pyplot.
# Con un backend interactivo (Qt/Tk), plt.show() bloquea el script
# esperando a que cierres la ventana manualmente. Como aquí solo
# necesitamos los PNG guardados con savefig(), Agg es lo correcto
# y además funciona igual en local, remoto o sin display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap.umap_ as umap

from tqdm import tqdm


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_PATH = Path(
    "/home/julianrgx/Escritorio/UG/LIDIA_8vo_Semestre/Proyecto_Integrador/data/VehicleClassificationDataset"
)


OUTPUT_PATH = Path(
    "./embedding_visualization"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

# Modelo utilizado para extraer embeddings
# Opciones:
# "resnet50"
# "efficientnet_b1"

MODEL_NAME = "resnet50"

# Tamaño de entrada
IMAGE_SIZE = 224

# Batch size
BATCH_SIZE = 64

# Workers
NUM_WORKERS = 4

# None = utilizar todas las imágenes
MAX_IMAGES_PER_CLASS = None

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")


# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

class VehicleDataset(Dataset):

    EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    def __init__(
        self,
        root,
        transform=None
    ):

        self.root = Path(root)
        self.transform = transform

        self.samples = []

        # Cada carpeta representa una clase
        class_dirs = sorted([
            d
            for d in self.root.iterdir()
            if d.is_dir()
        ])

        self.class_names = [
            d.name
            for d in class_dirs
        ]

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(
                self.class_names
            )
        }

        # ----------------------------------------------------
        # Cargar imágenes
        # ----------------------------------------------------

        for class_dir in class_dirs:

            class_name = class_dir.name

            class_idx = self.class_to_idx[
                class_name
            ]

            images = sorted([
                p
                for p in class_dir.rglob("*")
                if p.suffix.lower()
                in self.EXTENSIONS
            ])

            if MAX_IMAGES_PER_CLASS is not None:

                images = images[
                    :MAX_IMAGES_PER_CLASS
                ]

            for image_path in images:

                self.samples.append(
                    (
                        image_path,
                        class_idx
                    )
                )

        # ----------------------------------------------------
        # Estadísticas
        # ----------------------------------------------------

        print("\nDataset")
        print("-" * 50)

        for class_name, idx in (
            self.class_to_idx.items()
        ):

            count = sum(
                1
                for _, label
                in self.samples
                if label == idx
            )

            print(
                f"{class_name:20s}: {count}"
            )

        print("-" * 50)

        print(
            f"Total imágenes: "
            f"{len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, label = (
            self.samples[idx]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(
                image
            )

        return (
            image,
            label,
            str(image_path)
        )


# ============================================================
# MODELO
# ============================================================

def create_model():

    if MODEL_NAME == "resnet50":

        weights = (
            models.ResNet50_Weights.DEFAULT
        )

        model = models.resnet50(
            weights=weights
        )

        # Eliminar clasificador
        model.fc = nn.Identity()

    elif MODEL_NAME == "efficientnet_b1":

        weights = (
            models.EfficientNet_B1_Weights.DEFAULT
        )

        model = models.efficientnet_b1(
            weights=weights
        )

        # Eliminar clasificador
        model.classifier = nn.Identity()

    else:

        raise ValueError(
            f"Modelo desconocido: "
            f"{MODEL_NAME}"
        )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# EXTRAER EMBEDDINGS
# ============================================================

def extract_embeddings(
    model,
    dataloader
):

    all_embeddings = []
    all_labels = []
    all_paths = []

    with torch.no_grad():

        for (
            images,
            labels,
            paths
        ) in tqdm(
            dataloader,
            desc="Extrayendo embeddings"
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            embeddings = model(
                images
            )

            # Asegurar formato:
            # [batch, features]

            embeddings = embeddings.flatten(
                start_dim=1
            )

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

            all_paths.extend(
                paths
            )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    return (
        embeddings,
        labels,
        all_paths
    )


# ============================================================
# VISUALIZACIÓN 3D
# ============================================================

def plot_embedding_3d(
    embedding,
    labels,
    class_names,
    title,
    filename
):

    fig = plt.figure(
        figsize=(13, 10)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # --------------------------------------------------------
    # Una clase por grupo
    # --------------------------------------------------------

    for class_idx, class_name in enumerate(
        class_names
    ):

        mask = (
            labels == class_idx
        )

        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],

            label=class_name,

            alpha=0.65,

            s=15
        )

    # --------------------------------------------------------
    # Etiquetas
    # --------------------------------------------------------

    ax.set_title(
        title,
        fontsize=16
    )

    ax.set_xlabel(
        "Component 1"
    )

    ax.set_ylabel(
        "Component 2"
    )

    ax.set_zlabel(
        "Component 3"
    )

    ax.legend(
        bbox_to_anchor=(1.15, 1),
        loc="upper left"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------

    plt.savefig(
        OUTPUT_PATH / filename,
        dpi=300,
        bbox_inches="tight"
    )

    # plt.show() eliminado: con backend Agg no hace nada útil y con
    # un backend interactivo se queda esperando a que cierres la
    # ventana manualmente (esto es lo que probablemente causaba el
    # "congelamiento").

    plt.close(fig)

    print(
        f"Guardado: {OUTPUT_PATH / filename}",
        flush=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # DATASET
    # ========================================================

    print(
        "\nCreando dataset..."
    )

    dataset = VehicleDataset(
        DATASET_PATH,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True
    )

    # ========================================================
    # MODELO
    # ========================================================

    print(
        "\nCargando modelo..."
    )

    model = create_model()

    print(
        f"Modelo: {MODEL_NAME}"
    )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    embeddings, labels, paths = (
        extract_embeddings(
            model,
            dataloader
        )
    )

    print(
        "\nEmbedding shape:"
    )

    print(
        embeddings.shape
    )

    # A partir de aquí ya no necesitamos el modelo ni la GPU: todo
    # lo que sigue (PCA, t-SNE, UMAP) corre en CPU con sklearn.
    # Liberar el modelo evita que PyTorch siga reteniendo threads
    # que sklearn/OpenBLAS necesita.
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.set_num_threads(1)

    # Ejemplo:
    #
    # ResNet50:
    # (20562, 2048)
    #
    # EfficientNet-B1:
    # (20562, 1280)

    # ========================================================
    # GUARDAR EMBEDDINGS
    # ========================================================

    np.save(
        OUTPUT_PATH / "embeddings.npy",
        embeddings
    )

    np.save(
        OUTPUT_PATH / "labels.npy",
        labels
    )

    metadata = pd.DataFrame({

        "path": paths,

        "label": labels,

        "class": [
            dataset.class_names[label]
            for label in labels
        ]
    })

    metadata.to_csv(
        OUTPUT_PATH / "metadata.csv",
        index=False
    )

    # ========================================================
    # PCA → 3D
    # ========================================================

    print(
        "\nCalculando PCA 3D...",
        flush=True
    )

    t0 = time.time()

    pca = PCA(
        n_components=3,
        svd_solver="randomized",
        random_state=42
    )

    embedding_pca = (
        pca.fit_transform(
            embeddings
        )
    )

    print(
        f"PCA 3D listo en {time.time() - t0:.1f}s",
        flush=True
    )

    print(
        "\nVarianza explicada por PCA:"
    )

    print(
        pca.explained_variance_ratio_
    )

    print(
        "Varianza acumulada:",
        pca.explained_variance_ratio_.sum()
    )

    plot_embedding_3d(

        embedding_pca,

        labels,

        dataset.class_names,

        "PCA 3D - Vehicle Embeddings",

        "pca_3d.png"
    )

    # ========================================================
    # PCA → 50 → t-SNE → 3D
    # ========================================================

    print(
        "\nPreparando t-SNE...",
        flush=True
    )

    t0 = time.time()

    # --------------------------------------------------------
    # Reducción previa
    # --------------------------------------------------------
    #
    # Esto no es la visualización final.
    #
    # Se utiliza únicamente para reducir
    # el costo computacional de t-SNE.

    pca_tsne = PCA(
        n_components=min(
            50,
            embeddings.shape[1]
        )
    )

    embeddings_50 = (
        pca_tsne.fit_transform(
            embeddings
        )
    )

    print(
        f"Dimensión antes de t-SNE: "
        f"{embeddings_50.shape} "
        f"({time.time() - t0:.1f}s)",
        flush=True
    )

    # --------------------------------------------------------
    # t-SNE 3D
    # --------------------------------------------------------

    print(
        "\nCalculando t-SNE 3D... "
        "(puede tardar varios minutos con ~20k muestras)",
        flush=True
    )

    t0 = time.time()

    tsne = TSNE(

        n_components=3,

        perplexity=30,

        learning_rate="auto",

        init="pca",

        random_state=42,

        max_iter=1000,

        n_jobs=-1
    )

    embedding_tsne = (
        tsne.fit_transform(
            embeddings_50
        )
    )

    print(
        f"t-SNE 3D listo en {time.time() - t0:.1f}s",
        flush=True
    )

    plot_embedding_3d(

        embedding_tsne,

        labels,

        dataset.class_names,

        "t-SNE 3D - Vehicle Embeddings",

        "tsne_3d.png"
    )

    # ========================================================
    # UMAP → 3D
    # ========================================================

    print(
        "\nCalculando UMAP 3D...",
        flush=True
    )

    t0 = time.time()

    reducer = umap.UMAP(

        n_components=3,

        n_neighbors=15,

        min_dist=0.1,

        metric="cosine",

        random_state=42,

        n_jobs=1
        # nota: UMAP solo paraleliza (n_jobs=-1) si se omite
        # random_state; con random_state fijo (reproducibilidad)
        # sklearn/umap fuerza n_jobs=1 y lanza un warning si pones
        # otro valor.
    )

    embedding_umap = (
        reducer.fit_transform(
            embeddings
        )
    )

    print(
        f"UMAP 3D listo en {time.time() - t0:.1f}s",
        flush=True
    )

    plot_embedding_3d(

        embedding_umap,

        labels,

        dataset.class_names,

        "UMAP 3D - Vehicle Embeddings",

        "umap_3d.png"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "Proceso terminado correctamente."
    )

    print(
        "========================================"
    )

    print(
        f"\nResultados:"
    )

    print(
        OUTPUT_PATH.resolve()
    )


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":
    main()