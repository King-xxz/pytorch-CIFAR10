import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1=torch.nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        return self.model1(x)

#验证神经网络正确性
if __name__=="__main__":
    net = Net()
    input=torch.ones((64,3,32,32))
    output=net(input)
    print(output.shape)