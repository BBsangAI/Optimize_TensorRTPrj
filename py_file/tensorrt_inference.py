import tensorrt as trt

onnx_file_path = "gesture_7classification_model.onnx"
engine_file_path = "gesture_7classification_model.engine"

# 创建 TensorRT logger
logger = trt.Logger(trt.Logger.WARNING)

# 构建引擎
with trt.Builder(logger) as builder, \
        builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
        trt.OnnxParser(network, logger) as parser:
    # 读取 ONNX 模型文件
    with open(onnx_file_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print(f"Failed to parse the ONNX file: {onnx_file_path}")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()

    # 配置 builder
    print("start build")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace

    # 创建优化配置
    # profile = builder.create_optimization_profile()
    # profile.set_shape("input", (1, 3, 16, 112, 112), (1, 3, 16, 112, 112), (1, 3, 16, 112, 112))  # min/opt/max
    # config.add_optimization_profile(profile)
    
    # 构建 TensorRT 引擎
    with builder.build_serialized_network(network, config) as serialized_engine:
        with trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)

    # 保存引擎
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print(f"TensorRT engine saved to {engine_file_path}")


