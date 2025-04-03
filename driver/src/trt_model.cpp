# include <string>
# include <vector>
# include <memory>
# include <iostream>
# include "trt_timer.hpp"
# include "trt_model.hpp"
# include "trt_logger.hpp"
# include "trt_classifierInit.hpp"
# include "trt_calibration.hpp"
# include "NvInfer.h"
# include "NvOnnxParser.h"



namespace model{
Model::Model(std::string onnx_path, logger::Level level, model::Params params) {
    m_onnxPath = onnx_path;
    m_params = new Params(params);
    m_enginePath = getEnginePath(onnx_path, getPrec(params.prec));
    m_workspaceSize = WORKSPACESIZE;
    m_logger        = std::make_shared<logger::Logger>(level);
    m_timer         = std::make_shared<timer::Timer>();
}

void Model::load_image(std::string image_path){
    if(!fileExists(image_path)){
        LOGE("ERROR: %f not found\n", image_path);
    }else {
        m_imagePath = image_path;
        LOG("Model:         %s", getFileName(m_onnxPath).c_str());
        LOG("Image:         %s", getFileName(m_imagePath).c_str());
        LOG("Precision:     %s", getPrec(m_params->prec).c_str());
    }
}

void Model::init_model(){
    //engine检测 文件存在就不重复生成
    if(!fileExists(m_enginePath)){
        LOGV("%s not found. Building trt engine...", m_enginePath.c_str());
        build_engine();
    }
    else {
        LOGV("%s has been found. loading trt engine...", m_enginePath.c_str());
        load_engine();
    }
}

// 智能指针的生命周期仅局限于当前的{}, 因此在if外会被析构 -> 段错误
// 因此将其声明为全局变量
std::shared_ptr<Int8MinMaxCalibrator> calibrator_minmax;
std::shared_ptr<Int8EntropyCalibrator> calibrator_entropy;
bool Model::build_engine() {
    // 创建基本组件
    // auto builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*m_logger), model::destory_trt_ptr<nvinfer1::IBuilder>);
    // auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1), model::destory_trt_ptr<nvinfer1::INetworkDefinition>);
    //  auto config  = std::shared_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig(), model::destory_trt_ptr<nvinfer1::IBuilderConfig>);
    // auto parser  = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *m_logger), model::destory_trt_ptr<nvonnxparser::IParser>);
    // 设置参数

    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(*m_logger);
    // 创建网络定义
    const auto explicitBatch = 1U << static_cast<uint32_t>
        (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  // 显性批处理标志
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    // 创建ONNX解析器
    nvonnxparser::IParser * parser = nvonnxparser::createParser(*network, *m_logger);

    auto config  = std::shared_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig(), model::destory_trt_ptr<nvinfer1::IBuilderConfig>);
    config->setMaxWorkspaceSize(m_workspaceSize);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    if (!parser->parseFromFile(m_onnxPath.c_str(), 1)){
            return false;
        }
    // 加载成功
    printf("tensorRT load mask onnx model successfully!!!...\n");
    // printf("jie xi onnx....\n");
    // // parser解析onnx
    // if (!parser->parseFromFile(m_onnxPath.c_str(), 1)){
    //     return false;
    // }
    // else
    //     printf("tensorRT load mask onnx model successfully!!!...\n");

    /*INT8量化 INT8量化需要校准器，FP16不需要，直接config->setFlag(BuilderFlag::kFP16);*/
    # if 0
    // 检测当前平台是否支持int8计算
    if(builder->platformHasFastInt8() && m_params->prec == model::INT8){
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);  //精度优先
        
            if(m_params->cal == model::calibrator::Entropy){
            calibrator_entropy.reset(new Int8EntropyCalibrator(
                1,
                "calibration/calibration_list.txt",
                "calibration/calibration_table.txt",
                3 * 224 * 224, 224, 224));   // 自己的校准数据路径，输入大小以及宽高
            config->setInt8Calibrator(calibrator_entropy.get());
            }

            else if (m_params->cal == model::calibrator::MinMax) {
            calibrator_minmax.reset(new Int8MinMaxCalibrator(
                1,
                "calibration/calibration_list.txt",
                "calibration/calibration_table.txt",
                3 * 224 * 224, 224, 224));      // 自己的校准数据路径，输入大小以及宽高
            config->setInt8Calibrator(calibrator_minmax.get());
        }
    }
    # endif
    
    // 动态维度
    # if 1
    builder->setMaxBatchSize(1);
    auto input = network->getInput(0);
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        LOGE("ERROR: Failed to create profile");
        return false;
    }nvinfer1::Dims inputDims;
    inputDims.nbDims = 5;  // 5D 张量
    inputDims.d[0] = 1;    // batch_size
    inputDims.d[1] = 3;    // channels
    inputDims.d[2] = 16;   // depth (或时间步)
    inputDims.d[3] = 112;  // height
    inputDims.d[4] = 112;  // width
    
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, inputDims); //设置最小尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, inputDims); //设置优化尺寸
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, inputDims); //设置最大尺寸
    config->addOptimizationProfile(profile);
    # endif

    // 保存序列化后的engine
    printf("start build...\n");
    auto plan = builder->buildEngineWithConfig(*network, *config);
    nvinfer1::IHostMemory* model_stream = plan->serialize();
    save_plan(*model_stream);
    
    // 根据runtime初始化engine, context, 以及memory
    // 在build or load engine时, 直接分配好推理所需要的资源(主要就是bindings)
    setup(model_stream->data(), model_stream->size());
}

bool Model::load_engine() {
    if (!fileExists(m_enginePath)) {
        LOGE("engine does not exits! Program terminated");
        return false;
    }

    std::vector<unsigned char> modelData;
    modelData = loadFile(m_enginePath);
    // 根据runtime初始化engine, context, 以及memory
    printf("start setup\n");
    setup(modelData.data(), modelData.size());
    printf("finish setup\n");
    return true;
}

void Model::inference() {
    if (m_params->dev == CPU) {
        preprocess_cpu();
    } else {
        preprocess_gpu();
    }

    enqueue_bindings();

    if (m_params->dev == CPU) {
        postprocess_cpu();
    } else {
        postprocess_gpu();
    }
}

bool Model::enqueue_bindings() {
    m_timer->start_gpu();
    if(!m_context->enqueueV2((void**)m_bindings, m_stream, nullptr)){
        LOG("Error happens during inference part, program terminated");
        return false;
    }
    m_timer->stop_gpu();
    m_timer->duration_gpu("trt-inference(GPU)");
    return true;
}

// 保存序列化后的engine
void Model::save_plan(nvinfer1::IHostMemory& plan) {
    auto f = fopen(m_enginePath.c_str(), "wb");
    fwrite(plan.data(), 1, plan.size(), f);
    fclose(f);
}
// 获取精度
std::string Model::getPrec(model::precision prec) {
    switch(prec) {
        case model::precision::FP16:   return "fp16";
        case model::precision::INT8:   return "int8";
        default:                       return "fp32";
    }
}
}