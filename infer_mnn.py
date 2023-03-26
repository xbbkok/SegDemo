'''
 # @ Author: Ben.X
 # @ E-Mail: benx555@qq.com
 # @ Create Time: 2023-03-25 19:40:59
 # @ Description:  MNN-python模型预测
 '''

import MNN
import numpy as np
from PIL import Image

# pip install MNN
# https://pypi.org/project/MNN/


def get_session(modelPath):
    net = MNN.Interpreter(modelPath)
    net.setCacheFile("./cachefile")  

    net.setSessionMode(9)


    config = {}
    # config['backend'] = "OPENCL"
    # config['precision'] = "high"
    config['precision'] = 'low' 
    config['numThread'] = 0     
    config['backend'] = 0  
    session = net.createSession(config)

    return net, session


def crop(img):
    img = img.resize((320, 256), Image.NEAREST)
    return img

def get_input(img_path):

    image = Image.open(img_path)
    print(image.size)

    image = crop(image)
    image = np.array(image)
    image = image / 255.0
    image = image - (0.5, 0.5, 0.5)
    image = image / (0.5, 0.5, 0.5)

    image = image.transpose((2, 0, 1))
    image = image.astype(np.float32)

    tmp_input = MNN.Tensor((1, 3, 256,320), MNN.Halide_Type_Float,image, MNN.Tensor_DimensionType_Caffe)

    return tmp_input
                


def get_output(tmp_input ,net,session):

    inputTensor = net.getSessionInput(session)
    inputTensor.copyFrom(tmp_input)

    net.runSession(session)
    outputTensor = net.getSessionOutput(session)

    out = outputTensor.getNumpyData() #.astype(np.uint8)
    return out


def get_input2(img_path):
    import MNN.cv as cv
    import MNN.numpy as np
    import MNN.expr as expr
    img = cv.imread(img_path,cv.COLOR_BGR2RGB)

    img = img / 255.0
    img = (img - 0.5) / 0.5

    imgf = img.astype(np.float32)
    imgf_batch = np.expand_dims(imgf, 0)
    input_var = expr.convert(imgf_batch, expr.NCHW)
    input_var = MNN.Tensor(input_var)
    return input_var



if __name__ == '__main__':
    modelPath = './exported_models/SegModel.mnn'
    img_path = r"archive/images/0000051.jpg"

    # 第一步： 创建session
    net, session = get_session(modelPath)
    
    # 第二步：图片预处理 （两种方式都可以用，任选其一即可）
    tmp_input = get_input(img_path)
    # tmp_input = get_input2(img_path)

    # 第三步：模型预测获得输出结果
    output = get_output(tmp_input,net,session)

    # 第四步：输出结果后处理（图片上色）
    from process_data import gray2color
    gray2color(output,save_path='ouput_imgs/infer_mnn.png')
    
