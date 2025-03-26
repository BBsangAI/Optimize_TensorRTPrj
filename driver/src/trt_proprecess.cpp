# include "utils.hpp"
# include "trt_timer.hpp"
# include "trt_preprocess.hpp"
# include "opencv2/opencv.hpp"

namespace process{
void preprocess_resize_gpu(const cv::Mat &h_src, float* d_tar, const int &tarH, const int &tarW, \
                            float* h_mean, float* h_std, tactics tac){
    float*   d_mean = nullptr;
    float*   d_std  = nullptr;
    uint8_t* d_src  = nullptr;

    int height  = h_src.rows;
    int width   = h_src.cols;
    int channel = 3;

    int src_size  = height * width * channel * sizeof(uint8_t);
    int norm_size = 3 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_src, src_size));
    CUDA_CHECK(cudaMalloc(&d_mean, norm_size));
    CUDA_CHECK(cudaMalloc(&d_std, norm_size));

    CUDA_CHECK(cudaMemcpy(d_src, h_src.data, src_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean, h_mean, norm_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_std, h_std, norm_size, cudaMemcpyHostToDevice));

    // kernel function
    resize_gpu(d_src, d_tar, width, height, tarW, tarH, d_mean, d_std, tac);

    // host和device进行同步处理
    CUDA_CHECK(cuDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_std));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_src));

}
};