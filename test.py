import torch
import torchvision.transforms as transforms
from torch.autograd import Variable #自動微分用
from PIL import Image
import numpy as np
import model
from model import MODEL_PATH

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = model.Net()
net.cpu()
source = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
net.load_state_dict(source)

img = Image.open('car.jpg')
img = img.resize((32, 32))
img = transform(img).float()
img = Variable(img)
img = img.unsqueeze(0)

with torch.no_grad():
    outputs = net(img)

    #print(outputs)
    _, predicted = torch.max(outputs, 1)
    print(classes[predicted])
