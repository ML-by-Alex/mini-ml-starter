from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_loaders(data_dir: str, batch_size: int = 64, num_workers: int = 2):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
