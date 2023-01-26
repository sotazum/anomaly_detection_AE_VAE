from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))  # [0,1] => [-1,1]
])

class SplitedMNIST(Dataset):
    def __init__(self, labels, train=True):
        self.mnist = MNIST('./data', download=True,train=train, transform=transform)
        self.dataset = [self.mnist[i][0] for i in range(len(self.mnist)) if self.mnist[i][1] in labels]
        self.labels = labels
        # self.noise_label = noise_label

    def __len__(self):
        return int(len(self.dataset))

    def __getitem__(self, index):
        img = self.dataset[index]
        return img
