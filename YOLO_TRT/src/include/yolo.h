/*
 * @Version: v1.0
 * @Author: 東DONG
 * @Mail: cv_yang@126.com
 * @Date: 2023-01-16 09:08:20
 * @LastEditTime: 2023-01-31 15:47:16
 * @FilePath: /YOLO_TRT/src/include/yolo.h
 * @Description: 
 * Copyright (c) 2023 by ${東}, All Rights Reserved. 
 * 
 *    ┏┓　　　┏┓
 *  ┏┛┻━━━┛┻┓
 *  ┃　　　　　　　┃
 *  ┃　　　━　　　┃
 *  ┃　＞　　　＜　┃
 *  ┃　　　　　　　┃
 *  ┃...　⌒　...　┃
 *  ┃　　　　　　　┃
 *  ┗━┓　　　┏━┛
 *      ┃　　　┃　
 *      ┃　　　┃
 *      ┃　　　┃
 *      ┃　　　┃  神兽保佑
 *      ┃　　　┃  代码无bug　　
 *      ┃　　　┃
 *      ┃　　　┗━━━┓
 *      ┃　　　　　　　┣┓
 *      ┃　　　　　　　┏┛
 *      ┗┓┓┏━┳┓┏┛
 *        ┃┫┫　┃┫┫
 *        ┗┻┛　┗┻┛
 */

#ifndef YOLO_H
#define YOLO_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <cmath>

// #define DLL_EXPORT __declspec(dllexport)  

class Yolo_Det
{
public:
    int batchsize = 1;
    int input_c;
    int input_w;
    int input_h;
    int output_numbox;
    int output_numprob;
    int num_classes = 80;

    float conf_thresh = 0.25f;
    float nms_thresh = 0.5f;

    const std::string input_blob_name = "images";
    const std::string output_blob_name = "outputs";

    Yolo_Det(const std::string& _engine_file);
    virtual ~Yolo_Det();
    virtual void inference(cv::Mat& img);

private:
    int input_buffer_size;
    int output_buffer_size;
    int input_index;
    int output_index;

    float* host_input_cpu = nullptr;
    float* host_output_cpu = nullptr;
    float* device_input_gpu = nullptr;
    float* device_output_gpu = nullptr;

    float i2d[6], d2i[6];

    const std::string engine_file;
    cudaStream_t stream = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::ICudaEngine *engine_det = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;

    std::vector<void*> cudaOutputBuffer;
    std::vector<void*> hostOutputBuffer;

    void init_context();
    void destroy_context();
    void pre_process(cv::Mat& img);
    void post_process(cv::Mat& img);
};

#endif