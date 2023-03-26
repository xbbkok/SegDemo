'''
 # @ Author: Ben.X
 # @ E-Mail: benx555@qq.com
 # @ Create Time: 2023-03-25 13:18:33
 # @ Description: 原始数据处理
 '''

import os
import pandas as pd
import cv2
import numpy as np
import pprint
from PIL import Image
import pathlib
import random

def colorMask_2_grayMask(color_mask_path, sv_gray_path,all_colors):
    color_to_num_dic = {all_colors[k][0]:all_colors[k][-1] for k in all_colors.keys()}
    color = cv2.imread(color_mask_path)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    zeros = np.zeros(shape=(color.shape[0],color.shape[1]),dtype=np.uint8)

    for rgb in color_to_num_dic.keys():
        mask = np.all(color==np.array(rgb).reshape(1, 1, 3), axis=2)
        zeros[mask] = color_to_num_dic[rgb]

    cv2.imwrite(sv_gray_path, zeros)

def get_all_colors(csv_path='archive/labels_class_dict.csv'):
    df = pd.read_csv(csv_path)
    class_num_dic = {}
    for i in range(df.shape[0]):
        name = df.loc[i,'class_names']
        r = df.loc[i, 'r']
        g = df.loc[i, 'g']
        b = df.loc[i, 'b']
        class_num_dic[name] = [(r,g,b),i]
    pprint.pprint(class_num_dic,sort_dicts=False)
    '''
    {'sky': [(68, 1, 84), 0],
    'tree': [(72, 40, 140), 1],
    'road': [(62, 74, 137), 2],
    'grass': [(38, 130, 142), 3],
    'water': [(31, 158, 137), 4],
    'building': [(53, 183, 121), 5],
    'mountain': [(109, 205, 89), 6],
    'foreground': [(180, 222, 44), 7],
    'unknown': [(49, 104, 142), 8]}
    '''
    return class_num_dic


# cls_color_dic = {'sky': [(68, 1, 84), 0],
#     'tree': [(72, 40, 140), 1],
#     'road': [(62, 74, 137), 2],
#     'grass': [(38, 130, 142), 3],
#     'water': [(31, 158, 137), 4],
#     'building': [(53, 183, 121), 5],
#     'mountain': [(109, 205, 89), 6],
#     'foreground': [(180, 222, 44), 7],
#     'unknown': [(49, 104, 142), 8]}


cls_color_dic=get_all_colors()
def gray2color(img, save_path='my_mask.png'):
    '''
    from gray (num 1,2...25) to colorMask
    '''
    PALETTE= [item[0] for item in cls_color_dic.values()]

    lette = PALETTE
    img = np.uint8(img)
    print("shape xxx:",img.shape)
    img = Image.fromarray(img).convert("P")  
    palette = []
    for i in range(256):
        palette.extend((i, i, i))

    palette[:3*len(lette)] = np.array(
        lette, dtype='uint8').flatten()

    img.putpalette(palette)
    img.show()
    img.save(save_path)


def gray_2_color(img_path, save_path,all_colors):
    PALETTE = [item[0] for item in all_colors.values()]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    lette = PALETTE

    img = Image.fromarray(img)
    palette = []
    for i in range(256):
        palette.extend((i, i, i))

    palette[:3*len(lette)] = np.array(
        lette, dtype='uint8').flatten()

    img.putpalette(palette)
    img.save(save_path)

def demoTest():
    color_mask_path = 'archive/labels_colored/0000047.png'
    sv_gray_mask_path = 'gray_mask/0000047.png'
    sv_color_mask_path = 'color_mask/0000047.png'

    all_colors = get_all_colors()
    colorMask_2_grayMask(color_mask_path=color_mask_path, 
                 sv_gray_path=sv_gray_mask_path,
                 all_colors=all_colors)
    
    gray_2_color(img_path=sv_gray_mask_path, 
               save_path=sv_color_mask_path,
               all_colors=all_colors)

def cvt_gray_all(color_mask_dir,sv_gray_mask_dir):
    color_mask_dir = pathlib.Path(color_mask_dir)
    all_colors = get_all_colors()
    for colorMask in color_mask_dir.iterdir():
        sv_gray_mask_path = os.path.join(sv_gray_mask_dir,colorMask.name)
        colorMask_2_grayMask(color_mask_path=str(colorMask), 
                    sv_gray_path=sv_gray_mask_path,
                    all_colors=all_colors)


def split_train_test(mask_dir,rgb_dir,color_mask_dir):
    masks = os.listdir(mask_dir)
    
    random.shuffle(masks)
    set_train = random.sample(masks,int(len(masks)*0.9))
    set_test = set(masks) - set(set_train)
    def write_txt(sets,txt_path):
        with open(txt_path,'w') as f:
            for item in sets:
                rgb_path = os.path.join(rgb_dir, item.replace('png', 'jpg'))
                if  os.path.exists(rgb_path):

                    f.write(
                        rgb_path+ ',' + os.path.join(mask_dir,item) +  ',' +os.path.join(color_mask_dir,item)+ '\n'
                    )
                    f.flush()

    write_txt(set_train, 'train.txt')
    write_txt(set_test, 'test.txt')


if __name__ == '__main__':
    color_mask_dir = 'archive/labels_colored'
    sv_gray_mask_dir = 'gray_mask'
    rgb_dir = 'archive/images'
    color_mask_dir = 'archive/labels_colored'


    cvt_gray_all(color_mask_dir,sv_gray_mask_dir)
    split_train_test(sv_gray_mask_dir,rgb_dir,color_mask_dir)
