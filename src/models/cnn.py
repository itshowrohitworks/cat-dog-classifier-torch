import torch
import torch.nn as nn

class CatDogCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Convolution Layer:
        self.features = nn.Sequential(
            
            # input shape: 3 x 224 x 224

            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )
        """
        till here the output shape: 
        batch_size, 128, 1, 1
        """


        # Linear Layer: Flatten Layer
        self.classifier = nn.Sequential(

            nn.Flatten(),
            # output shape: one vector of length 128

            nn.Linear(in_features=128,out_features=128),
            nn.ReLU(),

            # Dropout: regularization technique -> to prevent overfitting during training
            nn.Dropout(p=0.5),

            nn.Linear(in_features=128,out_features=2)
        )
        """
        till here the output:
        2 Classes: 
        Cat and Dog
        """
    
    def forward(self,x):

        x = self.features(x)

        x = self.classifier(x)

        return x