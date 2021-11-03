# should run on any dataset like the mnist set?
# training script
import torch as t
from torch.nn.modules import loss
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss
from model import CovNet
from tqdm import tqdm


def train(model, device, train_loader, val_loader, loss_fn, optim, n_epoch, learning_rate):
    loss_fn = loss_fn()
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

        train_acc = evaluate(model, data_loader=train_loader, device=device)

        if val_loader is not None:
            val_accuracy = evaluate(
                model, data_loader=val_loader, device=device)
        else:
            val_accuracy = None

        pbar.set_description(
            f'For epoch number {epoch+1}, Training Acc: {train_acc}, Validation Acc: {val_accuracy} Loss: {l}')
        pbar.update()

    return model, val_accuracy


def evaluate(model, device, data_loader):
    model.eval()

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
    image_resize_shape = (28, 28)

    # Hyperparameters
    n_epoch = 15
    batch_size = 100
    learning_rate = 0.01

    # Load train images into dataset
    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_resize_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_set = tv.datasets.ImageFolder(
        train_images_dir, transform=transformer)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Define net
    model = CovNet(image_size=image_resize_shape)
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    model = model.to(device)

    loss_fn = CrossEntropyLoss
    optim = Adam

    # train loader and model to device
    model, _ = train(model=model, train_loader=train_loader, val_loader=None,
                     loss_fn=loss_fn, optim=optim, n_epoch=n_epoch, device=device,
                     learning_rate=learning_rate)

    model_save_path = './models/fashion_mnst_cnn.bin'
    t.save(model.state_dict(), model_save_path)
