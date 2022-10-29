# dataset

import torch
import torchvision
import torchvision.transforms as transforms

class CIFAR10():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    classes = classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    @staticmethod
    def load_cifar_100(root="./data", batch_size=4, shuffle=True, num_workers=2):
        
        trainloader = CIFAR100.load_train_cifar_10(root=root, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        testloader = CIFAR100.load_test_cifar_10(root=root, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
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


if __name__ == "__main__":
    CIFAR10.load_cifar_10()