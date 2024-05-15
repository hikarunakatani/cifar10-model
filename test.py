import os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import model
from model import MODEL_PATH


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    net = model.CNN()
    net.cpu()
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        net.load_state_dict(state_dict)
        print("Model loaded successfully.")
    else:
        print("No model checkpoint found at the specified path.")

    img = Image.open("bird.jpg")
    img = img.resize((32, 32))
    img = transform(img).float()
    img = Variable(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        outputs = net(img)

        # print(outputs)
        _, predicted = torch.max(outputs, 1)
        print(classes[predicted])


if __name__ == "__main__":
    main()
