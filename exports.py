'''
 # @ Author: Ben.X
 # @ E-Mail: benx555@qq.com
 # @ Create Time: 2023-03-25 18:05:11
 # @ Description: 导出模型
 '''

import torch
from net import SegModel2
import coremltools as ct

def export_onnx(torch_model, example_input):
    torch.onnx.export(torch_model, example_input, "exported_models/SegModel.onnx", verbose=True, input_names=['input'], output_names=['output'])

def export_coreml(traced_model, example_input):
    model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(
        bias=[-1, -1, -1],
            scale=1.0 /(255.0 * 0.5),
            shape=example_input.shape,
            color_layout="RGB",
            channel_first=True
            )],
    )

    model.save("exported_models/SegModel.mlmodel")

def export_torchScirpt(traced_model):
    torch.jit.save (traced_model, "exported_models/SegModel.pt")
    

if __name__ == '__main__':


    torch_model = SegModel2()

    device = torch.device('cpu')
    ckpt_path = r'ckpts/net.pth'


    torch_model.to(device)
    torch_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    torch_model.eval()
    
    example_input = torch.rand(1, 3, 256, 320) 
    traced_model = torch.jit.trace(torch_model, example_input)

    export_onnx(torch_model,example_input)
    export_coreml(traced_model, example_input)
    export_torchScirpt(traced_model)