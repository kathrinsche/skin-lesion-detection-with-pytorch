import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes=7, pretrained=True, freeze_backbone=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
