import torch
from PIL import Image
from torch.utils.data import Dataset

class Vimeo90K(Dataset):
    def __init__(self, data_root, subset, ratio=0.95):
        self.data_root = f'{data_root}/sequences/'

        if subset == 'train' or subset == 'val':
            train_txt = f'{data_root}/tri_trainlist.txt'
            with open(train_txt) as f:
                train_file_names = f.read().splitlines()
            split = int(len(train_file_names) * ratio)
            if subset == 'train':
                self.file_names = train_file_names[:split]
            else:
                self.file_names = train_file_names[split:]
        
        elif subset == 'test':
            test_txt = f'{data_root}/tri_testlist.txt'
            with open(test_txt) as f:
                self.file_names = f.read()/splitlines()

        else:
            raise ValueError("Invalid subset name")

    def __len__(self):
        return(len(self.file_names))

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image_root = f'{self.data_root}/{file_name}/'
        
        img1 = torch.tensor(Image.open(f'{image_root}/im1.png')) / 255
        img2 = torch.tensor(Image.open(f'{image_root}/im2.png')) / 255
        img3 = torch.tensor(Image.open(f'{image_root}/im3.png')) / 255
        t = 0.5

        return img1, img3, t, img2
        