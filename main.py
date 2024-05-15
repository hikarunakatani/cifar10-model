import argparse
import torch
import os
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import model
from model import MODEL_PATH
import aws_action


def train(lr, momentum, num_epochs, env):
    """train the CNN model

    Args:
        lr (float): learning rate used in model training
        momentum (float): momentum used in model training
        num_epochs (int): number of epochs used in model training
    """

    download_flag = False

    # Dataset directory
    data_dir = "./data"

    # Download training data if it doesn't exist
    if not os.path.exists(os.path.join(data_dir, "cifar-10-batches-py")):
        if env == "local":
            download_flag = True
        elif env == "ecs":
            aws_action.download_data()

    # Preprocess data
    # Transform PIL Image to tensor
    # Normalize a tensor image in each channel with mean and standard deviation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Downaload training data of CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=download_flag, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=os.cpu_count()
    )

    # Download validation data of CIFAR-10
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=download_flag, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=os.cpu_count()
    )

    # Combine as dictionary
    dataloaders_dict = {"train": trainloader, "val": testloader}

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

    # Add logs to Tensorboard
    writer = SummaryWriter("logs")

    # Use GPU if there are available resources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.CNN().to(device)

    # Load pre-trained model if exists
    if os.path.exists(MODEL_PATH):
        source = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
        net.load_state_dict(source)

    # Define loss function and optimization method
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    num_epochs = num_epochs

    for epoch in range(num_epochs):  # Loop for the specified number of times
        print(f"Epoch {epoch+1}/{num_epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0

            for input_data, label in tqdm(dataloaders_dict[phase]):
                input_data, label = input_data.to(device), label.to(
                    device
                )  # Enable GPU
                optimizer.zero_grad()

                # Compute gradient in training phase
                with torch.set_grad_enabled(phase == "train"):
                    predicted_label = net(input_data)
                    loss = criterion(predicted_label, label)
                    _, pred_index = torch.max(predicted_label, axis=0)
                    _, label_index = torch.max(label, axis=0)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # Update loss summary
                    epoch_loss += loss.item() * input_data.size(0)
                    # Update the number of correct prediction
                    epoch_corrects += torch.sum(pred_index == label_index)

            # show loss and accuracy for each epoch
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects / len(dataloaders_dict[phase].dataset)

            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

            # record the loss and accuracy
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"Accuracy/{phase}", epoch_acc, epoch)

    print("Finished Training")
    writer.close()
    torch.save(net.state_dict(), MODEL_PATH)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument(
        "--env", type=str, default="local", help="executing environment"
    )
    args = parser.parse_args()

    train(lr=args.lr, momentum=args.momentum, num_epochs=args.num_epochs, env=args.env)
    aws_action.upload_model()


if __name__ == "__main__":
    main()
