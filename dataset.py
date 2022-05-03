import torch
import imageio
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as TF

class Vimeo90K(Dataset):
    def __init__(self, data_root, subset, ratio=0.95, crop_size=(224,224)):
        self.data_root = f'{data_root}/sequences/'
        self.subset = subset
        self.crop_size = crop_size
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
                self.file_names = f.read().splitlines()

        else:
            raise ValueError("Invalid subset name")

    def __len__(self):
        return(len(self.file_names))

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image_root = f'{self.data_root}/{file_name}/'
        
        img1 = np.array(imageio.imread(f'{image_root}/im1.png')).transpose(2,0,1) / 255
        img2 = np.array(imageio.imread(f'{image_root}/im2.png')).transpose(2,0,1) / 255
        img3 = np.array(imageio.imread(f'{image_root}/im3.png')).transpose(2,0,1) / 255
        
        if self.subset == 'test':
            return torch.tensor(img1), torch.tensor(img3), torch.tensor(img2)

        # random crop 
        _, ih, iw = img1.shape
        h, w = self.crop_size
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img1 = img1[:, x:x+h, y:y+w]
        img2 = img2[:, x:x+h, y:y+w]
        img3 = img3[:, x:x+h, y:y+w]

        if self.subset == 'train':
            # v flip
            if np.random.rand() < 0.5:
                img1 = img1[:, ::-1, :]
                img2 = img2[:, ::-1, :]
                img3 = img3[:, ::-1, :]
            # h flip
            if np.random.rand() < 0.5:
                img1 = img1[:, :, ::-1]
                img2 = img2[:, :, ::-1]
                img3 = img3[:, :, ::-1]
            # reverse
            if np.random.rand() < 0.5:
                tmp = img1
                img1 = img3
                img3 = tmp
            # to tensor
            img1, img2, img3 = torch.tensor(img1.copy()), torch.tensor(img2.copy()), torch.tensor(img3.copy())
            # rotate
            p = np.random.rand()
            if p < 0.25:
                img1 = TF.functional.rotate(img1, 90)
                img2 = TF.functional.rotate(img2, 90)
                img3 = TF.functional.rotate(img3, 90)
            elif p < 0.5:
                img1 = TF.functional.rotate(img1, 180)
                img2 = TF.functional.rotate(img2, 180)
                img3 = TF.functional.rotate(img3, 180)
            elif p < 0.75:
                img1 = TF.functional.rotate(img1, 270)
                img2 = TF.functional.rotate(img2, 270)
                img3 = TF.functional.rotate(img3, 270)

        return img1, img3, img2

if __name__ == '__main__':
    dataset = Vimeo90K('../data/vimeo_triplet', 'train')
    for i in range(10):
        i1, i2, i3 = dataset[i]
        print(i1.shape)