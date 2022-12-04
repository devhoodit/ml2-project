# dataset

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy as np

class CIFAR10():
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    @staticmethod
    def load_cifar_10(root="./data", batch_size=4, shuffle=True, num_workers=2):
        
        trainloader = CIFAR10.load_train_cifar_10(root=root, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        testloader = CIFAR10.load_test_cifar_10(root=root, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
        return trainloader, testloader

    @staticmethod
    def load_train_cifar_10(root="./data", batch_size=4, shuffle=True, num_workers=2):
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=CIFAR10.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return trainloader
    
    @staticmethod
    def load_test_cifar_10(root="./data", batch_size=4, shuffle=True, num_workers=2):
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=CIFAR10.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return testloader

class CIFAR100():
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    @staticmethod
    def load_cifar_100(root="./data", batch_size=4, shuffle=True, num_workers=2):
        
        trainloader = CIFAR100.load_train_cifar_100(root=root, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        testloader = CIFAR100.load_test_cifar_100(root=root, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
        return trainloader, testloader

    @staticmethod
    def load_train_cifar_100(root="./data", batch_size=4, shuffle=True, num_workers=2):
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=CIFAR100.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return trainloader
    
    @staticmethod
    def load_test_cifar_100(root="./data", batch_size=4, shuffle=True, num_workers=2):
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=CIFAR10.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return testloader

class DataAugDataset(Dataset):
    def __init__(self, root="./data", train=True) -> None:
        dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        dataset_size = len(dataset)
        x = dataset.data[np.random.choice(dataset_size, size=dataset_size)]
        
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        
        x = list(map(transform, x))
        self.x = x
        self.y = np.zeros(len(x), dtype=np.int8)
        
        x = dataset.data[np.random.choice(dataset_size, size=dataset_size)]

        data_augmentation = T.Compose(
            [
                T.ToTensor(),
                T.RandomErasing(p=1),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        x = list(map(data_augmentation, x))
        y = np.ones(len(x), dtype=np.int8)
        
        self.x += x
        self.y = np.concatenate((self.y, y), axis=None)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], torch.IntTensor([self.y[idx]])
    

class CIFAR10DataAug():
    @staticmethod
    def load_train(root="./data", batch_size=4, shuffle=True, num_workers=2):
        trainset = DataAugDataset(root=root)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return trainloader

    @staticmethod
    def load_test(root="./data", batch_size=4, shuffle=True, num_workers=2):
        testset = DataAugDataset(root=root, train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return testloader

if __name__ == "__main__":
    # CIFAR10.load_cifar_10()
    # CIFAR100.load_cifar_100()
    CIFAR10DataAug.load_train()
    