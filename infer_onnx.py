'''
 # @ Author: Ben.X
 # @ E-Mail: benx555@qq.com
 # @ Create Time: 2023-03-25 19:54:09
 # @ Description: onnx-python 模型预测
 '''

import numpy as np
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import onnxruntime as ort
import torchvision.transforms as transforms

x_transform =  transforms.Compose([
                        transforms.ToTensor(),  # -> [0,1]
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
                        ])

def pre_process(img):
    img = x_transform(img)
    inputs = torch.stack([img], dim=0)
    return inputs

def predict(onnx_path, img_path):

    
    # 1.图像预处理
    img = Image.open(img_path)
    img = img.resize((320,256), Image.NEAREST)
    inputs = pre_process(img)

    # 2.加载会话
    sess = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider']) #'TensorrtExecutionProvider', 'CUDAExecutionProvider', 

    input_name = sess.get_inputs()[0].name

    # 3.预测推理
    result = sess.run([], {input_name: np.array(inputs)})

    # 4.输出结果的后处理
    result = np.array(result)
    from process_data import gray2color
    result = result.squeeze(0)
    gray2color(result,save_path='ouput_imgs/infer_onnx.png')

if __name__ == '__main__':
    onnx_path = r'exported_models/SegModel.onnx'
    img_path = r"archive/images/0000051.jpg"
    predict(onnx_path,img_path)