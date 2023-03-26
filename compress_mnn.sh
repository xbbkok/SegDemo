mnnconvert -f ONNX \
--modelFile exported_models/SegModel.onnx \
--MNNModel exported_models/SegModel_fp16.mnn  \
--bizCode biz
-- fp16


mnnconvert -f ONNX \
--modelFile exported_models/SegModel.onnx \
--MNNModel exported_models/SegModel_bit8.mnn  \
--weightQuantBits 8