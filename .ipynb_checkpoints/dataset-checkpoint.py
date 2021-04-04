import torch
from torchvision import transforms, datasets

def get_data_loader(batch_size=128):
    ''' Get MNIST dataset with normalization '''
    trainset = datasets.MNIST("./data", train=True, download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )

    train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size=batch_size, shuffle=True)
    return train_loader