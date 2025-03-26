# include <stdio.h>
# include <iostream>
# include "cuda_runtime_api.h"
# include "trt_preprocess.hpp"


__global__ void nearest_BGR2RGB_nhwc2nchw_norm_kernel( float* tar, const uint8_t* src, 
    int tarW, int tarH, 
    int srcW, int srcH,
    float scaled_w, float scaled_h,
    float* d_mean, float* d_std) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int src_y = floor((float)y * scaled_h);
    int src_x = floor((float)x * scaled_w);

    if(src_x<0 || src_y<0 || src_x>src_W || src_y>srcH){
    }else{
        int tarIdx  = y * tarW + x;
        int tarArea = tarW * tarH;

        // nearest neighbour -- 计算src中最近邻坐标的索引
        int srcIdx = (src_y * srcW + src_x) * 3;

        // nearest neighbour -- 实现nearest beighbour的resize + BGR2RGB + nhwc2nchw + norm
        tar[tarIdx + tarArea * 0] = (src[srcIdx + 2] / 255.0f - d_mean[2]) / d_std[2];
        tar[tarIdx + tarArea * 1] = (src[srcIdx + 1] / 255.0f - d_mean[1]) / d_std[1];
        tar[tarIdx + tarArea * 2] = (src[srcIdx + 0] / 255.0f - d_mean[0]) / d_std[0];
    }
}

void resize_gpu(const uint8_t* d_src, float* d_tar, 
    int srcW, int srcH, int tarW, int tarH, 
    float* d_mean, float* d_std, process::tactics tac)
{
    dim3 dimBlock(32,32,1);
    dim3 dimGrid(tarW/ 32+1, tarH /32+1, 1);

    //scaled resize
    float scaled_h = (float)srcH / tarH;
    float scaled_w = (float)srcW / tarW;
    float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);

    nearest_BGR2RGB_nhwc2nchw_norm_kernel
    <<<dimGrid, dimBlock>>>(d_tar, d_src, tarW, srcW,srcH, scaled_w, scaled_h, d_mean, d_std);

}