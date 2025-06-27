from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "model_save/ecg_model_quantized.onnx",
    "model_save/ecg_model_quantized_int8.onnx",
    weight_type=QuantType.QInt8
)