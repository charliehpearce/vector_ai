from train import train
import torchvision as tv
import torch as t
from itertools import product
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss
from model import CovNet
import numpy as np


def hyperparameter_combinations(tuning_dict):
    keys = tuning_dict.keys()
    params = [tuning_dict[key] for key in keys]
    d_out = [dict(zip(keys, i)) for i in product(*params)]
    return d_out


if __name__ == "__main__":
    # Load data into dataloaders and split into train validation sets
    train_images_dir = './data/train/png/'
    image_resize_shape = (28, 28)

    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_resize_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    data_set = tv.datasets.ImageFolder(
        train_images_dir, transform=transformer)

    # Train validation split
    val_size = 0.2
    n_val_examples = round(len(data_set)*val_size)
    n_train_examples = len(data_set)-n_val_examples
    train_set, val_set = random_split(
        data_set, [n_train_examples, n_val_examples])

    # Create tuning grid with different hyperparameters
    tuning_grid = {'learning_rate': [0.001, 0.01],
                   'n_epoch': [5, 10, 15],
                   'batch_size': [100, 500, 1000]}

    params = hyperparameter_combinations(tuning_grid)

    # Loop over model training and get validation accuracy
    # This could be fairly easily run in paralel
    accuracies = []

    for i, p in enumerate(params):
        print(f'Trial {i+1}/{len(params)}')
        # Hyperparameters
        n_epoch = p['n_epoch']
        batch_size = p['batch_size']
        learning_rate = p['learning_rate']

        # Load into dataloaders
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        # Define net and send to GPU if avalible
        model = CovNet(image_size=image_resize_shape)
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        model = model.to(device)

        loss_fn = CrossEntropyLoss()
        optim = Adam

        # train loader and model to device
        model, acc = train(model=model, train_loader=train_loader, val_loader=val_loader,
                           loss_fn=loss_fn, optim=optim, n_epoch=n_epoch, device=device,
                           learning_rate=learning_rate)
        accuracies.append(acc)

    print(params[np.argmax(accuracies)])
