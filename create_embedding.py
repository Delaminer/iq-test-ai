import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
# Load the pretrained model
model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')
# Set model to evaluation mode
model.eval()
resize = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def embed(img):
    img = np.array(img)
    if len(img.shape) == 2:
        # add the channel dimension
        img = np.stack((img,)*3, axis=-1)
    img = to_tensor(img)
    if img.dtype != torch.float32:
        img = img.type(torch.float32)
    img = resize(img)
    img = img[:3]
    img = normalize(img)
    img = img.unsqueeze(0)
    # Initialize the embedding tensor
    my_embedding = torch.zeros(512, dtype=torch.float32)

    # Hook to extract the features from the avgpool layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())  # Flatten the output

    hook = layer.register_forward_hook(copy_data)

    # Forward pass to compute the embedding
    with torch.no_grad():
        model(img)

    hook.remove()  # Remove the hook
    return my_embedding