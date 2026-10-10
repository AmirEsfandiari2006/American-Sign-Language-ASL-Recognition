import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=29):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=128 * 8 * 8, out_features=256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        return self.network(x)