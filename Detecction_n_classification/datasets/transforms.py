from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ============================================================
# CONFIGURACIÓN
# ============================================================

IMAGE_SIZE = 240


# Normalización utilizada por los modelos preentrenados
# de ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# RESIZE MANTENIENDO ASPECT RATIO + PADDING
# ============================================================

class ResizeAndPad:
    """
    Redimensiona una imagen manteniendo su aspect ratio
    y posteriormente agrega padding negro hasta obtener
    una imagen cuadrada de target_size x target_size.

    La imagen queda centrada dentro del canvas.

    Ejemplo:

        800 x 600
             ↓
        240 x 180
             ↓
        padding negro
             ↓
        240 x 240

    También permite hacer upscaling de imágenes pequeñas.

    Ejemplo:

        47 x 92
          ↓
        123 x 240
          ↓
        padding negro
          ↓
        240 x 240
    """

    def __init__(
        self,
        target_size=IMAGE_SIZE,
        allow_upscale=True
    ):
        self.target_size = target_size
        self.allow_upscale = allow_upscale

    def __call__(self, image):

        # ----------------------------------------------------
        # Asegurar RGB
        # ----------------------------------------------------

        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Dimensiones inválidas: {width}x{height}"
            )

        # ----------------------------------------------------
        # Calcular escala
        # ----------------------------------------------------

        scale = self.target_size / max(width, height)

        # Si no queremos hacer upscale y la imagen es menor
        # que target_size, mantenemos su tamaño original.
        if not self.allow_upscale and max(width, height) < self.target_size:
            scale = 1.0

        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))

        # ----------------------------------------------------
        # Resize
        # Hacemos resize de la imagen manteniendo su aspect ratio
        # y el largura o altura máxima será target_size.
        # 
        # ----------------------------------------------------
        
        if new_width != width or new_height != height:

            image = image.resize(
                (new_width, new_height),
                Image.Resampling.BICUBIC
            )

        # ----------------------------------------------------
        # Crear canvas negro
        # ----------------------------------------------------

        canvas = Image.new(
            "RGB",
            (self.target_size, self.target_size),
            (0, 0, 0)
        )

        # ----------------------------------------------------
        # Centrar imagen
        # ----------------------------------------------------

        x = (self.target_size - new_width) // 2
        y = (self.target_size - new_height) // 2

        canvas.paste(image, (x, y))

        return canvas


# ============================================================
# TRANSFORMS DE ENTRENAMIENTO
# ============================================================

def get_train_transforms(image_size=IMAGE_SIZE, allow_upscale=True):
    """
    Transformaciones utilizadas durante entrenamiento.

    Primero:
        Resize + Padding

    Después:
        Data Augmentation

    Finalmente:
        Tensor + Normalización ImageNet
    """

    return T.Compose([

        ResizeAndPad(
            target_size=image_size,
            allow_upscale=allow_upscale
        ),

        # ----------------------------------------------------
        # DATA AUGMENTATION
        # ----------------------------------------------------

        T.RandomHorizontalFlip(
            p=0.5
        ),

        T.RandomRotation(
            degrees=8
        ),

        T.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),

        # ----------------------------------------------------
        # PIL → Tensor
        # ----------------------------------------------------

        T.ToTensor(),

        # ----------------------------------------------------
        # Normalización ImageNet
        # ----------------------------------------------------

        T.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


# ============================================================
# TRANSFORMS DE VALIDACIÓN
# ============================================================

def get_val_transforms(
    image_size=IMAGE_SIZE,
    allow_upscale=True
):
    """
    Transformaciones para validation.

    NO contiene augmentations aleatorias.
    """

    return T.Compose([

        ResizeAndPad(
            target_size=image_size,
            allow_upscale=allow_upscale
        ),

        T.ToTensor(),

        T.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


# ============================================================
# TRANSFORMS DE TEST
# ============================================================

def get_test_transforms(
    image_size=IMAGE_SIZE,
    allow_upscale=True
):
    """
    Transformaciones para test.

    Debe ser determinista.
    """

    return T.Compose([

        ResizeAndPad(
            target_size=image_size,
            allow_upscale=allow_upscale
        ),

        T.ToTensor(),

        T.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])