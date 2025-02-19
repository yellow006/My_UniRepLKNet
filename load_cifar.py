import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar_dataset(batch_size ,classes):
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ])
    if classes == 100:
        train_dataset = datasets.CIFAR100(root='./cifar100/', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root='./cifar100/', train=False, download=True, transform=test_transform)
    else:
        train_dataset = datasets.CIFAR10(root='./cifar10/', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='./cifar10/', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

if __name__ == '__main__':
    # test
    train, test = get_cifar_dataset(batch_size=64, classes=10)
    for img, lb in train:
        print("images shape: ", img.shape)
        print("labels shape: ", lb.shape)


