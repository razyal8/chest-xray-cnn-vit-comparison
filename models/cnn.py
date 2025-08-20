# models/cnn.py
import torch
import torch.nn as nn
import torchvision.models as models

def build_cnn(name: str = "resnet18", num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
    name = name.lower()
    if name == "custom":
        from .custom_cnn import SmallCNN
        net = SmallCNN(num_classes=num_classes)
        return net

    if name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, num_classes)
        backbone = net
    elif name == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, num_classes)
        backbone = net
    elif name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, num_classes)
        backbone = net
    elif name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_f, num_classes)
        backbone = net
    else:
        raise ValueError(f"Unknown CNN model: {name}")

    if freeze_backbone:
        for n, p in backbone.named_parameters():
            if "fc" in n or "classifier.1" in n:
                continue
            p.requires_grad = False

    return backbone
