from datasets.vehicle_dataset import VehicleClassificationDataset
from datasets.transforms import get_train_transforms


dataset = VehicleClassificationDataset(
    root_dir="VehicleClassificationDataset",
    transform=get_train_transforms()
)

print("Número de imágenes:", len(dataset))
print("Clases:", dataset.classes)

image, label = dataset[0]

print("Shape:", image.shape)
print("Label:", label)
print("Clase:", dataset.classes[label])