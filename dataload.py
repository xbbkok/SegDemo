'''
 # @ Author: Ben.X
 # @ E-Mail: benx555@qq.com
 # @ Create Time: 2023-03-24 08:33:05
 # @ Description: create dataloader
 '''

import torch.utils.data as data
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import PIL.Image as Image
import os
import numpy as np
import torch

class SegData(Dataset):
    def __init__(self, train_file_path, val_file_path, data_root, train=True):
        super().__init__()
        if train:
            self.files = open(train_file_path,'r')
        else:
            self.files  = open(val_file_path,'r')  

        self.train_data = []
        self.val_data = []

        self.data=[]
        for line in self.files.readlines():
            img, label = line.strip().split(',')[0:2]
            img_path = os.path.join(data_root, img)
            label_path = os.path.join(data_root, label)

            self.data.append([img_path,label_path])

        self.x_transform =  transforms.Compose([
                        transforms.ToTensor(),  # -> [0,1]
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
                        ])


    def __len__(self,):
        return len(self.data)

    def __getitem__(self, index):
        x_path, y_path = self.data[index]

        # print(x_path)
        # print(y_path)

        img_x = Image.open(x_path)    
        img_y = Image.open(y_path)

        img_x = img_x.resize((320,256),Image.NEAREST)
        img_y = img_y.resize((320,256),Image.NEAREST)

        img_x = self.x_transform(img_x)

        img_y = np.array(img_y) # PIL -> ndarry
        img_y = torch.from_numpy(img_y).long() 
        return img_x, img_y

if __name__ == '__main__':
    train_file_path = 'train.txt'
    val_file_path = 'test.txt'
    data_root = os.getcwd()

    dataset = SegData(train_file_path=train_file_path,
                        val_file_path=val_file_path,
                        data_root=data_root,
                        train=True)
    
    my_loader = DataLoader(dataset=dataset,
                            batch_size=4,
                            shuffle=True,
                            drop_last=False,
                            num_workers=0)
    for i , (x, y) in enumerate(my_loader):
        print(x.shape)
