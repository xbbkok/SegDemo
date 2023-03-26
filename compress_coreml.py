'''
 # @ Author: Ben.X
 # @ E-Mail: benx555@qq.com
 # @ Create Time: 2023-03-26 16:31:54
 # @ Modified time: 2023-03-26 16:32:02
 # @ Description: coreml 模型压缩
 '''

import coremltools as ct
from coremltools.models.neural_network import quantization_utils

# load full precision model
model_fp32 = ct.models.MLModel('exported_models/SegModel.mlmodel')


'''Quantizing to float 16, which reduces by half the model's disk size, 
is the safest quantization option since it generally 
does not affect the model's accuracy:'''
model_fp16 = quantization_utils.quantize_weights(model_fp32, nbits=16)
model_fp16.save('exported_models/SegModel_fp16.mlmodel')


# quantize to 8 bit using linear mode
model_8bit = quantization_utils.quantize_weights(model_fp32, nbits=8)
model_8bit.save("exported_models/SegModel_bit8_linear.mlmodel")
print("linear success")

# quantize to 8 bit using LUT kmeans mode
model_8bit = quantization_utils.quantize_weights(model_fp32, nbits=8,
                             quantization_mode="kmeans")
model_8bit.save("exported_models/SegModel_kmeans.mlmodel")
print("kmeans success")


# quantize to 8 bit using linearsymmetric mode
model_8bit = quantization_utils.quantize_weights(model_fp32, nbits=8,
                             quantization_mode="linear_symmetric")
model_8bit.save("exported_models/SegModel_linear_symmetric.mlmodel")
print("linear-symmetric success")