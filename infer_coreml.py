'''
 # @ Author: Ben.X
 # @ E-Mail: benx555@qq.com
 # @ Create Time: 2023-03-25 19:49:29
 # @ Description: coreml-python-macos预测
 '''


import coremltools as ct
import numpy as np
import PIL.Image
import torch
from PIL import Image
import torch.nn.functional as F
import time

if __name__ == '__main__':
    # 第一步：加载模型
    model = ct.models.MLModel('exported_models/SegModel.mlmodel')

    # 第二步：打开图片
    img = PIL.Image.open('archive/images/0000051.jpg')
    img = img.resize((320, 256),Image.NEAREST)

    # 第三步：模型预测
    t1 = time.time()
    out_dict = model.predict({'x': img})
    print("time cost: ", time.time()-t1)

    results = out_dict[list(out_dict.keys())[0]]

    # 第四步：后处理(图片上色)
    # post_process(results)

    from process_data import gray2color
    gray2color(results,save_path='ouput_imgs/infer_coreml.png')