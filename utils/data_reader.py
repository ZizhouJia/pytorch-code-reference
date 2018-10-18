import torchvision
import torch.utils.data as Data
import data_argument


def get_cifar10_data_loader(batch_size=32,transform=data_argument.get_transforms_without_argument()):
    trainset=torchvision.datasets.CIFAR10(root='../../dataset_folder/cifar10',
                                 train=True,
                                 download=True,
                                 transform=transform)

    train_loader=Data.DataLoader(trainset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=0)

    testset=torchvision.datasets.CIFAR10(root='../../dataset_folder/cifar10',
                                 train=False,
                                 download=False,
                                 transform=data_argument.get_transforms_without_argument())

    test_loader=Data.DataLoader(testset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=0)
    return train_loader,test_loader

def get_mnist_data_loader(batch_size=32):
    trainset=torchvision.datasets.MNIST(root='../../dataset_folder/mnist',
                                 train=True,
                                 download=True,
                                 transform=torchvision.transforms.ToTensor())

    train_loader=Data.DataLoader(trainset,
                                        batch_size=32,
                                        shuffle=True,
                                        num_workers=0)

    testset=torchvision.datasets.MNIST(root='../../dataset_folder/mnist',
                                 train=False,
                                 download=True,
                                 transform=torchvision.transforms.ToTensor())

    test_loader=Data.DataLoader(testset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=0)
    return train_loader,test_loader
