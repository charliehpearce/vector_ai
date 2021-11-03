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


def evaluate(model, device, data_loader):
    model.eval()

    with t.no_grad():
        # eval on val_loader
        n_correct = 0
        n_samples = 0
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = t.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (preds == labels).sum().item()
        accuracy = 100*n_correct/n_samples
    return accuracy


if __name__ == "__main__":
    train_images_dir = './data/test/png/'
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

    test_set = tv.datasets.ImageFolder(
        train_images_dir, transform=transformer)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Define net
    model = CovNet(image_size=image_resize_shape)
    model.load_state_dict(t.load('./models/fashion_mnst_cnn.bin'))
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(evaluate(model=model, device=device, data_loader=test_loader))
    # Result: 87.87% accuracy on test set
