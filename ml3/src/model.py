import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST:
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 5, 3, padding='same')
            self.conv2 = nn.Conv2d(5, 9, 3, padding='same')
            self.conv3 = nn.Conv2d(9, 13, 3, padding='same')
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(7*7*13, 200)
            self.fc2 = nn.Linear(200, 10)
            
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = F.relu(x)
            
            x = torch.flatten(x, 1)
            
            x = self.fc1(x)
            x = F.relu(x)
            
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            
            return output
    
    class BNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 5, 3, padding='same')
            self.conv2 = nn.Conv2d(5, 9, 3, padding='same')
            self.conv3 = nn.Conv2d(9, 13, 3, padding='same')
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(7*7*13, 500)
            self.fc2 = nn.Linear(500, 300)
            self.fc3 = nn.Linear(300, 100)
            self.fc4 = nn.Linear(100, 10)
            
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = F.relu(x)
            
            x = torch.flatten(x, 1)
            
            x = self.fc1(x)
            x = F.relu(x)
            
            x = self.fc2(x)
            x = F.relu(x)
            
            x = self.fc3(x)
            x = F.relu(x)
            
            x = self.fc4(x)
            output = F.log_softmax(x, dim=1)
            
            return output
        
    class MNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 5, 3, padding='same')
            self.conv2 = nn.Conv2d(5, 9, 3, padding='same')
            self.conv3 = nn.Conv2d(9, 13, 3, padding='same')
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(7*7*13, 500)
            self.fc2 = nn.Linear(500, 400)
            self.fc3 = nn.Linear(400, 300)
            self.fc4 = nn.Linear(300, 150)
            self.fc5 = nn.Linear(150, 10)
            
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = F.relu(x)
            
            x = torch.flatten(x, 1)
            
            x = self.fc1(x)
            x = F.relu(x)
            
            x = self.fc2(x)
            x = F.relu(x)
            
            x = self.fc3(x)
            x = F.relu(x)
            
            x = self.fc4(x)
            x = F.relu(x)
            
            x = self.fc5(x)
            output = F.log_softmax(x, dim=1)
            
            return output
    
    class CNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 5, 3, padding='same')
            self.conv2 = nn.Conv2d(5, 9, 3, padding='same')
            self.conv3 = nn.Conv2d(9, 13, 3, padding='same')
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(7*7*13, 200)
            self.fc2 = nn.Linear(200, 10)
            
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = F.relu(x)
            
            x = torch.flatten(x, 1)
            
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = x + torch.sin(x*0.01) * 0.01
            output = F.log_softmax(x, dim=1)
            
            return output
        
class CIFAR10:
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
    class BNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 250)
            self.fc2 = nn.Linear(250, 150)
            self.fc3 = nn.Linear(150, 80)
            self.fc4 = nn.Linear(80, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x    

    class BNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 300)
            self.fc2 = nn.Linear(300, 200)
            self.fc3 = nn.Linear(200, 124)
            self.fc4 = nn.Linear(124, 60)
            self.fc5 = nn.Linear(60, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = self.fc5(x)
            return x
