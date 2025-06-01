import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


# print(torch.__version__)  # PyTorch version
# print(torch.version.cuda)  # CUDA version compatible with PyTorch
# print(torch.cuda.is_available()) 

def get_model(device):
    # model = models.resnet50(pretrained=True)
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Binary classification
    return model.to(device)

def get_loss():
    return nn.BCEWithLogitsLoss()  # Sigmoid + binary cross-entropy

def get_optimizer(model, learning_rate=0.001):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)