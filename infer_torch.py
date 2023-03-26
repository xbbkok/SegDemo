'''
 # @ Author: Ben.X
 # @ E-Mail: benx555@qq.com
 # @ Create Time: 2023-03-25 19:07:52 
 # @ Description: pytorch预测
 '''

import torch
from net import SegModel
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from process_data import gray2color



def pre_process(img):
    x_transform =  transforms.Compose([
                        transforms.ToTensor(),  # -> [0,1]
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
                        ])
    img = img.resize((320, 256))
    img = x_transform(img)
    inputs = torch.stack([img], dim=0)
    return inputs


def predict(ckpt_path,img_path):
    # 1.加载模型
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    net = SegModel()
    net.to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()

    # 2.数据预处理
    img = Image.open(img_path)
    img = pre_process(img)

    in_data = img.to(device)

    # 3.预测输出结果
    out_data = net(in_data)

    # 4.输出结果后处理
    y = F.softmax(out_data, dim=1) 
    y = torch.argmax(y, dim=1)
    y = torch.squeeze(y).numpy() 
    gray2color(y,save_path='ouput_imgs/infer_torch.png')


if __name__ == '__main__':
    ckpt_path = r'ckpts/net.pth'
    img_path = r"archive/images/0000051.jpg"
    predict(ckpt_path,img_path)