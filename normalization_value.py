import torch
from torch.utils.data import Dataset ,DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(244),
    transforms.ToTensor()
])

path = 'D:\current_work\Pytorch_image_classification\pizza-not-pizza\pizza_not_pizza'

dataset = ImageFolder(path,transform=transform)
print('Dataset created')
print(f'dataset length {len(dataset)}')
dataloader = DataLoader(dataset,batch_size=1000)
print('dataloader created')
data = next(iter(dataloader))
print('calculating')
print(f'mean {data[0].mean()}')
print(f'std {data[0].std()}')
