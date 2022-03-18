import torch
import torch.nn
from torchsummary import summary

if __name__ == '__main__':
    device = torch.device("cpu")
    model = torch.load('tests/mobilenetv2_model.pt', map_location=device)
    summary(model, (3, 224, 224), device=device)

    # print(model)
