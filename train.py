'''
 # @ Author: Ben.X
 # @ E-Mail: benx555@qq.com
 # @ Create Time: 2023-03-25 17:39:44
 # @ Description: 模型训练
 '''


from dataload import SegData
from net import SegModel
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import os


def train():

    train_file_path = 'train.txt'
    val_file_path = 'test.txt'
    data_root = os.getcwd()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    net = SegModel()
    net.to(device)

    batch_size = 2
    epochs = 100

    save_ckpt_path = './ckpts/net.pth'
    if os.path.exists(save_ckpt_path):
        net.load_state_dict(torch.load(save_ckpt_path, map_location=device))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    dataset = SegData(train_file_path=train_file_path,
                        val_file_path=val_file_path,
                        data_root=data_root,
                        train=True)
    my_loader = DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           drop_last=False,
                           num_workers=0)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(my_loader):
            in_data = x.to(device)
            labels = y.to(device)

            out_data = net(in_data)

            loss = loss_fn(out_data, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i >= 100 and (i % 100 == 0):
                torch.save(net.state_dict(), save_ckpt_path)

            if i % 10 == 0:
                print(loss.item())
        print(f"epoch : {epoch} , loss :{loss.item()}")

if __name__ == '__main__':
    train()