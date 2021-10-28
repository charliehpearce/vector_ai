# should run on any dataset like the mnist set?
# training script
import torch as t
from torch import functional
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss
from torchvision.transforms.functional import invert
from torchvision.transforms.transforms import Grayscale
from tqdm.std import trange
from model import CovNet
from tqdm import tqdm


def train(model, device, train_loader, val_loader, loss_fn, optim, n_epoch, learning_rate):
    optim = optim(model.parameters(), lr=learning_rate)
    pbar = tqdm(total=n_epoch)

    for epoch in range(n_epoch):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optim.zero_grad()

            # Forward pass
            outputs = model(images)

            # Backprop
            l = loss_fn(outputs, labels)
            l.backward()
            optim.step()

        val_accuracy = evaluate(model, data_loader=val_loader, device=device)
        train_acc = evaluate(model, data_loader=train_loader, device=device)

        pbar.set_description(
            f'For epoch number {epoch}, Training Acc: {train_acc}, Validation Acc: {val_accuracy} Loss: {l}')
        pbar.update()

    return model


def evaluate(model, device, data_loader):
    with t.no_grad():
        # eval on val_loader
        n_correct = 0
        n_samples = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = t.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (preds == labels).sum().item()
        accuracy = 100*n_correct/n_samples
    return accuracy


if __name__ == "__main__":
    train_images_dir = './data/train/png/'

    # Hyperparameters
    n_epoch = 10
    batch_size = 1000
    learning_rate = 0.001

    # Load train images into dataset
    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        # this does assume the size is
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_set = tv.datasets.ImageFolder(
        train_images_dir, transform=transformer)
    label_map = train_set.class_to_idx

    val_size = 0.1
    # train validation split
    n_val_examples = round(len(train_set)*val_size)
    n_train_examples = len(train_set)-n_val_examples
    train_set, val_set = random_split(
        train_set, [n_train_examples, n_val_examples])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Define net and p
    model = CovNet()
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    model = model.to(device)

    loss_fn = CrossEntropyLoss()
    optim = Adam

    # train loader and model to device
    train(model=model, train_loader=train_loader, val_loader=val_loader,
          loss_fn=loss_fn, optim=optim, n_epoch=n_epoch, device=device,
          learning_rate=learning_rate)
