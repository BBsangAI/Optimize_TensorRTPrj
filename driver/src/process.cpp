#include "process.hpp"
#include <cuda_runtime.h>
#include <iostream>

namespace process {

// 实现GPU图像预处理的函数定义
bool preprocess_resize_gpu(const cv::Mat& input, void* output, 
                          int target_height, int target_width,
                          float* mean, float* std, tactics method) {
    // 调用批处理版本的重载函数
    return preprocess_resize_gpu(input, static_cast<float*>(output), 
                                target_height, target_width,
                                mean, std, method);
}

bool preprocess_resize_gpu(const cv::Mat& input, float* output, 
                          int target_height, int target_width,
                          float* mean, float* std, tactics method) {
    try {
        // 调整图像大小
        cv::Mat resized;
        cv::resize(input, resized, cv::Size(target_width, target_height));
        
        // 转换为RGB并归一化 (简化版，实际应该在GPU上完成)
        cv::Mat normalized;
        resized.convertTo(normalized, CV_32FC3, 1.0/255.0);
        
        // 将图像数据从HWC格式转换为CHW格式，并应用均值和标准差
        // 注意：这里只是简化的CPU实现，实际应该在GPU上实现
        
        // 分配临时内存
        cv::Mat chw(target_height * target_width * 3, 1, CV_32FC1);
        float* chwPtr = (float*)chw.data;
        
        // HWC转CHW并归一化
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < target_height; ++h) {
                for (int w = 0; w < target_width; ++w) {
                    float pixel = normalized.at<cv::Vec3f>(h, w)[c];
                    // 应用均值和标准差
                    pixel = (pixel - mean[c]) / std[c];
                    chwPtr[c * target_height * target_width + h * target_width + w] = pixel;
                }
            }
        }
        
        // 将处理好的数据复制到输出
        // 注意：实际应该直接在GPU上处理，避免中间拷贝
        cudaMemcpy(output, chwPtr, sizeof(float) * target_height * target_width * 3, cudaMemcpyHostToDevice);
        
        return true;
    }
    catch(const std::exception& e) {
        std::cerr << "图像预处理错误: " << e.what() << std::endl;
        return false;
    }
}

} // namespace process
