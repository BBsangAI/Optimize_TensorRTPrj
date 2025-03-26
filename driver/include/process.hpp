#pragma once
#include <opencv2/opencv.hpp>

namespace process {
    
    enum tactics {
        GPU_BILINEAR,
        GPU_NEAREST,
        // 其他类型
    };
    
    // 单张图像预处理
    bool preprocess_resize_gpu(const cv::Mat& input, void* output, 
                              int target_height, int target_width,
                              float* mean, float* std, tactics method);
    
    // 针对指针偏移的批处理版本
    bool preprocess_resize_gpu(const cv::Mat& input, float* output, 
                              int target_height, int target_width,
                              float* mean, float* std, tactics method);
}
