import torch.nn as nn
import torchvision.models as models

def build_cnn(name: str = "resnet18", num_classes: int = 2, pretrained: bool = True):
    name = name.lower()
    if name == "custom":
        from .custom_cnn import CustomCNN
        return CustomCNN(num_classes=num_classes)
    if name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, num_classes)
        return net
    else:
        raise ValueError(f"Unknown CNN model: {name}")
