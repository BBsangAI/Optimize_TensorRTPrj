#include "image_labels.hpp"
# include "trt_model.hpp"
# include "trt_classifierInit.hpp"
# include "NvInfer.h"
# include "NvOnnxParser.h"

// classifier类任务的初始化相关内容
// input、output、bindings、分配host/device memory

namespace model{
namespace classifier{
void Classifier::setup(const void* data, const std::size_t size){  
    //创建基本组件（runtime, engine, context）
    m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*m_logger), model::destory_trt_ptr<nvinfer1::IRuntime>);  // shared_ptr,使用自己构建的删除器（destory_trt_ptr）来释放指针
    m_engine  = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(data, size), model::destory_trt_ptr<nvinfer1::ICudaEngine>);
    m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext(), model::destory_trt_ptr<nvinfer1::IExecutionContext>);
    m_inputDims   = m_context->getBindingDimensions(0);// 动态维度
    m_outputDims  = m_context->getBindingDimensions(1);// 动态维度
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    m_inputSize     = m_params->img.t * m_params->img.h * m_params->img.w * m_params->img.c * sizeof(float);
    m_outputSize    = m_params->num_cls * sizeof(float);
    m_imgArea       = m_params->img.h * m_params->img.w;
    printf("%d * %d * %d * %d\n", m_params->img.t, m_params->img.h, m_params->img.w, m_params->img.c);
     // malloc
    // 0: device   1: host
    CUDA_CHECK(cudaMallocHost(&m_inputMemory[0], m_inputSize));
    CUDA_CHECK(cudaMallocHost(&m_outputMemory[0], m_outputSize));
    CUDA_CHECK(cudaMalloc(&m_inputMemory[1], m_inputSize));
    CUDA_CHECK(cudaMalloc(&m_outputMemory[1], m_outputSize));

    // 创建bindings
    m_bindings[0] = m_inputMemory[1];
    m_bindings[1] = m_outputMemory[1];
}

void Classifier::load_batch(const std::vector<cv::Mat>& images) {
    m_batchImages = images;
}

bool Classifier::preprocess_cpu(){
    /*Preprocess -- 获取mean， std（图像归一化的参数）*/
    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};

    /*Preprocess -- 读取数据*/
    // cv::Mat input_image;
    // input_image = cv::imread(m_imagePath);
    if (m_batchImages.empty()) {
        LOGE("ERROR: Image file not founded! Program terminated"); 
        return false;
    }

    /*Preprocess -- -- host端进行normalization和BGR2RGB, NHWC->NCHW*/
      for (int batch = 0; batch < m_params->img.t; batch++) {
            // 计算当前批次索引的内存偏移
        cv::resize(m_batchImages[batch], m_batchImages[batch], cv::Size(m_params->img.w, m_params->img.h), 0, 0, cv::INTER_LINEAR);
        /*Preprocess -- host端进行normalization和BGR2RGB, NHWC->NCHW*/
        int index;
        int offset_ch0 = m_imgArea * 0;
        int offset_ch1 = m_imgArea * 1;
        int offset_ch2 = m_imgArea * 2;
        for (int i = 0; i < m_inputDims.d[4]; i++) {
            for (int j = 0; j < m_inputDims.d[1]; j++) {       //1 3 16 112 112
                index = i * m_inputDims.d[1] * m_inputDims.d[3] + j * m_inputDims.d[3];
                m_inputMemory[0][offset_ch2++] = (m_batchImages[batch].data[index + 0] / 255.0f - mean[0]) / std[0];
                m_inputMemory[0][offset_ch1++] = (m_batchImages[batch].data[index + 1] / 255.0f - mean[1]) / std[1];
                m_inputMemory[0][offset_ch0++] = (m_batchImages[batch].data[index + 2] / 255.0f - mean[2]) / std[2];
            }
        }

        // 保存每张处理后的图像
        // std::string savePath = "../OptmiziePrj/test_image/processed_image_" + std::to_string(i) + ".jpg";  // 保存路径和文件名
        // cv::Mat savedImage;
        // cv::cvtColor(m_batchImages[i], savedImage, cv::COLOR_RGB2BGR);  // 转换颜色通道，从 RGB 转回 BGR
        // if (!cv::imwrite(savePath, savedImage)) {
        //     std::cerr << "图像保存失败：" << savePath << std::endl;
        // } else {
        //     std::cout << "保存成功: " << savePath << std::endl;
        // }
    }

    /*Preprocess -- 测速*/
    m_timer->start_cpu();
    /*Preprocess -- 把host的数据移动到device上*/
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

    m_timer->stop_cpu();
    m_timer->duration_cpu<timer::Timer::ms>("preprocess(CPU)");

    return true;
}

bool Classifier::preprocess_gpu() {
    /*Preprocess -- 获取mean, std*/
    float mean[]       = {0.406, 0.456, 0.485};
    float std[]        = {0.225, 0.224, 0.229};

    // 如果有批次图像，处理批次
    if (!m_batchImages.empty()) {
        m_timer->start_gpu();
        
        // 这里处理批次图像
        // 假设我们已经有一个处理批次的GPU函数
        for (int i = 0; i < m_batchImages.size() && i < m_params->img.t; i++) {
            // 计算当前批次索引的内存偏移
            float* currentInput = static_cast<float*>(m_inputMemory[1]) + i * m_imgArea * m_params->img.c;
            
            // 处理每张图像
            process::preprocess_resize_gpu(m_batchImages[i], m_inputMemory[1],
                                         m_params->img.h, m_params->img.w, 
                                         mean, std, process::tactics::GPU_BILINEAR);
        }
        
        m_timer->stop_gpu();
        m_timer->duration_gpu("preprocess_batch(GPU)");
        return true;
    } else if (!m_imagePath.empty()) {
        // 单张图像处理，使用原有代码
        cv::Mat input_image;
        input_image = cv::imread(m_imagePath);
        if (input_image.data == nullptr) {
            LOGE("ERROR: file not founded! Program terminated"); return false;
        }

        m_timer->start_gpu();
        
        process::preprocess_resize_gpu(input_image, m_inputMemory[1],
                                     m_params->img.h, m_params->img.w, 
                                     mean, std, process::tactics::GPU_BILINEAR);

        m_timer->stop_gpu();
        m_timer->duration_gpu("preprocess(GPU)");
        return true;
    }
    
    return false;
}

bool Classifier::postprocess_cpu(){
    /*Postprocess -- 测速*/
    m_timer->start_cpu();
    /*Postprocess -- 将device上的数据移动到host上*/
    int output_size    = m_params->num_cls * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[0], m_inputMemory[1], output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    /*Postprocess -- 寻找label*/
    ImageNetLabels labels;
    int pos = max_element(m_inputMemory[0], m_inputMemory[0] + m_params->num_cls) - m_inputMemory[0];
    float confidence = m_inputMemory[0][pos] * 100;
    
    m_timer->stop_cpu();
    m_timer->duration_cpu<timer::Timer::ms>("postprocess(CPU)");

    LOG("Inference result: %s", labels.imagenet_labelstring(pos).c_str());   
    LOG("Confidence is %.3f%%\n", confidence);   
    return true;
}

bool Classifier::postprocess_gpu() {
    /*
        由于classification task的postprocess比较简单，所以CPU/GPU的处理这里用一样的
        对于像yolo这种detection model, postprocess会包含decode, nms这些处理。可以选择在CPU还是在GPU上跑
    */
    return postprocess_cpu();

}

std::shared_ptr<Classifier> make_classifier(std::string onnx_path, logger::Level level, model::Params params)
{
    auto classifier = make_shared<Classifier>(onnx_path, level, params);
    classifier->init_model();    // 创建即初始化
    return classifier;
}

};
};