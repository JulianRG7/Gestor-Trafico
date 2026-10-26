from torchvision.models import (
    efficientnet_b1,
    EfficientNet_B1_Weights,
)

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)

model_effnet = efficientnet_b1()
model_resnet = resnet50()

print("EfficientNet-B1 model:")
print(model_effnet)
# print("\nResNet-50 model:")
# print(model_resnet)
