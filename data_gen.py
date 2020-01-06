from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from os.path import join

img_size = 256
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((img_size + 30, img_size + 30)),
    transforms.RandomCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, train_or_test='train'):
        super(DatasetFromFolder, self).__init__()
        self.root = image_dir
        self.trans = train_transform if train_or_test == 'train'else test_transform
        self.roota = join(self.root, train_or_test) + 'A'
        self.rootb = join(self.root, train_or_test) + 'B'
        self.image_a = [x for x in listdir(self.roota)]
        self.image_b = [x for x in listdir(self.rootb)]

    def __getitem__(self, index):
        a = Image.open(join(self.roota, self.image_a[index])).convert('RGB')
        b = Image.open(join(self.rootb, self.image_b[index])).convert('RGB')
        a = a.resize((img_size, img_size), Image.BICUBIC)
        b = b.resize((img_size, img_size), Image.BICUBIC)

        a = self.trans(a)
        b = self.trans(b)
        return a, b

    def __len__(self):
        return len(self.image_a)


if __name__ == '__main__':
    dataset = DatasetFromFolder('data')
    # print(dataset[0])
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in loader:
        print(data[0].shape)
        data1 = (data[0].squeeze().permute(1, 2, 0) * std + mean) * 255
        data1 = data1.float().numpy().astype(np.uint8)
        image_pil = Image.fromarray(data1)
        image_pil.show()

        data2 = (data[1].squeeze().permute(1, 2, 0) * std + mean) * 255
        data2 = data2.float().numpy().astype(np.uint8)
        image_pil = Image.fromarray(data2)
        image_pil.show()
