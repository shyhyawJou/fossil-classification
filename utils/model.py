from torch import nn
from torchvision.models import mobilenet_v3_small



def load_model(device):
    model = mobilenet_v3_small(True)
    model.classifier[3] = nn.Linear(1024, 16)
    model = model.to(device)
    return model
