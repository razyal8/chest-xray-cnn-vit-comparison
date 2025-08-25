import torch.nn as nn

def conv_bn(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class CustomCNN(nn.Module):
    def __init__(self, num_classes=2, drop=0.2):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(3, 32),
            conv_bn(32, 32),
            nn.MaxPool2d(2),          

            conv_bn(32, 64),
            conv_bn(64, 64),
            nn.MaxPool2d(2),         

            conv_bn(64, 128),
            conv_bn(128, 128),
            nn.MaxPool2d(2),         

            conv_bn(128, 256),
            nn.MaxPool2d(2),        
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
