from torch import nn


class CovNet(nn.Module):
    # something wrong with sizing
    def __init__(self) -> None:
        super(CovNet, self).__init__()

        self.cov_layer1 = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, 6, 5),  # n channels, output channels, kernel
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.cov_layer2 = nn.Sequential(
            # Second conv layer
            nn.Conv2d(6, 12, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(12*4*4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 60),
            nn.ReLU(inplace=True),
            nn.Linear(60, 10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.cov_layer1(x)
        x = self.cov_layer2(x)
        x = x.view(-1, 12*4*4)  # flatten
        x = self.fc(x)
        return x
