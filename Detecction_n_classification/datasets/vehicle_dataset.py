# datasets/vehicle_dataset.py
# datasets/vehicle_dataset.py

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# CLASES
# ============================================================

CLASS_NAMES = [
    "Ambulance",
    "Ambulance-no-emergency",
    "Car",
    "Cargo",
    "FireTruck",
    "FireTruck-no-emergency",
    "Motorcycle",
    "Passenger",
    "Police",
    "Police-no-emergency",
]


CLASS_TO_IDX = {
    class_name: idx
    for idx, class_name in enumerate(CLASS_NAMES)
}


# ============================================================
# EXTENSIONES DE IMAGEN
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# DATASET
# ============================================================

class VehicleClassificationDataset(Dataset):
    """
    Dataset para clasificación de vehículos.

    Estructura esperada:

        VehicleClassificationDataset/
        ├── Ambulance/
        ├── Ambulance-no-emergency/
        ├── Car/
        ├── Cargo/
        ├── FireTruck/
        ├── FireTruck-no-emergency/
        ├── Motorcycle/
        ├── Passenger/
        ├── Police/
        └── Police-no-emergency/

    Cada muestra devuelve:

        image, label

    image:
        Tensor [C, H, W]

    label:
        entero entre 0 y 9
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        samples=None,
    ):
        """
        Parameters
        ----------
        root_dir : str o Path
            Directorio raíz del dataset.

        transform : callable, optional
            Transformaciones aplicadas a las imágenes.

        samples : list, optional
            Lista de muestras previamente seleccionadas.

            Cada elemento debe tener la forma:

                (image_path, label)

            Si es None, se escanean todas las imágenes.
        """
        print(root_dir)
        self.root_dir = Path(root_dir)
        self.transform = transform

        # ----------------------------------------------------
        # Verificar directorio
        # ----------------------------------------------------
        print(f"Verificando directorio del dataset: {self.root_dir}")
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"No existe el directorio del dataset:\n"
                f"{self.root_dir}"
            )

        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"La ruta no es un directorio:\n"
                f"{self.root_dir}"
            )

        # ----------------------------------------------------
        # Clases
        # ----------------------------------------------------

        self.classes = CLASS_NAMES.copy()

        self.class_to_idx = CLASS_TO_IDX.copy()

        # ----------------------------------------------------
        # Samples
        # ----------------------------------------------------

        if samples is None:
            self.samples = self._load_samples()
        else:
            self.samples = list(samples)

        # ----------------------------------------------------
        # Verificar dataset
        # ----------------------------------------------------

        if len(self.samples) == 0:
            raise RuntimeError(
                "El dataset no contiene imágenes."
            )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        self.labels = [
            label
            for _, label in self.samples
        ]

        # ----------------------------------------------------
        # Image paths
        # ----------------------------------------------------

        self.image_paths = [
            image_path
            for image_path, _ in self.samples
        ]

    # ========================================================
    # CARGAR TODAS LAS IMÁGENES
    # ========================================================

    def _load_samples(self):
        """
        Busca las imágenes de todas las clases.

        Returns
        -------
        list
            Lista de:

                (image_path, label)
        """

        samples = []

        for class_name in self.classes:

            class_dir = self.root_dir / class_name

            # ------------------------------------------------
            # Verificar clase
            # ------------------------------------------------

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"No existe la carpeta de la clase:\n"
                    f"{class_dir}"
                )

            # ------------------------------------------------
            # Buscar imágenes
            # ------------------------------------------------

            image_paths = sorted(
                path
                for path in class_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            )

            label = self.class_to_idx[class_name]

            # ------------------------------------------------
            # Agregar muestras
            # ------------------------------------------------

            for image_path in image_paths:

                samples.append(
                    (
                        image_path,
                        label,
                    )
                )

        return samples

    # ========================================================
    # LEN
    # ========================================================

    def __len__(self):
        return len(self.samples)

    # ========================================================
    # GETITEM
    # ========================================================

    def __getitem__(self, index):
        """
        Carga una imagen y aplica el transform correspondiente.
        """

        image_path, label = self.samples[index]

        # ----------------------------------------------------
        # Cargar imagen
        # ----------------------------------------------------

        try:

            with Image.open(image_path) as image:

                # --------------------------------------------
                # Convertir siempre a RGB
                # --------------------------------------------

                image = image.convert("RGB")

                # --------------------------------------------
                # Aplicar transform
                # --------------------------------------------

                if self.transform is not None:
                    image = self.transform(image)

        except Exception as error:

            raise RuntimeError(
                f"Error al cargar la imagen:\n"
                f"{image_path}\n"
                f"Error: {error}"
            ) from error

        return image, label