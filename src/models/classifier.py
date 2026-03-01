# src/models/classifier.py
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights

def build_model(num_classes: int):
    # 모든 파라미터를 학습 가능하게 설정
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)
    n_inputs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(n_inputs, num_classes)
    
    return model