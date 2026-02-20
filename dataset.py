from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFAR10Dataset(Dataset):
    """
    Custom wrapper around torchvision CIFAR-10.
    This satisfies the requirement of implementing our own Dataset class.
    """

    def __init__(self, root: str, train: bool):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=self.transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y