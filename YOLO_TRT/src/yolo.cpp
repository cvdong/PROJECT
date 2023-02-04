/*
 * @Version: v1.0
 * @Author: 東DONG
 * @Mail: cv_yang@126.com
 * @Date: 2023-01-16 09:08:20
 * @LastEditTime: 2023-01-31 14:45:06
 * @FilePath: /YOLO_TRT/src/yolo.cpp
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

#include <chrono>
#include <iostream>
#include <fstream> 
#include <math.h>
#include "include/yolo.h"
#include "include/utils.h"

static const int DEVICE  = 0;


Yolo_Det::Yolo_Det(const std::string& _engine_file):engine_file(_engine_file){
    std::cout<<"---engine_file---: " << engine_file << std::endl;
    init_context();
}


Yolo_Det::~Yolo_Det()
{
    destroy_context();
    std::cout << "------------------destroyed all !!!----------------" << std::endl;
}


void Yolo_Det::init_context(){
    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    
    size_t size{0};
    std::ifstream file(engine_file, std::ios::binary);

    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    TRTLogger logger;
    // trt initializer
    runtime = nvinfer1::createInferRuntime(logger);
    assert(runtime != nullptr);

    engine_det = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine_det != nullptr); 
    
    context = engine_det->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    
    // get parms
    input_index = engine_det->getBindingIndex(input_blob_name.c_str());
    output_index = engine_det->getBindingIndex(output_blob_name.c_str());
    assert(input_index==0);
    assert(output_index==1);
    auto input_dims = engine_det->getBindingDimensions(input_index);
    auto output_dims = engine_det->getBindingDimensions(output_index);
    
    input_c = input_dims.d[1];
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];

    input_buffer_size = batchsize * input_c * input_h * input_w * sizeof(float);
    
    // cudaHostAlloc cudaMallocHost cudaMalloc
    CHECK(cudaHostAlloc((void**)&host_input_cpu, input_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc((void**)&device_input_gpu, input_buffer_size));

    output_numbox = output_dims.d[1];
    output_numprob = output_dims.d[2];
    
    output_buffer_size = batchsize * output_numbox * output_numprob * sizeof(float);
 
    CHECK(cudaHostAlloc((void**)&host_output_cpu, output_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc((void**)&device_output_gpu, output_buffer_size));

    CHECK(cudaStreamCreate(&stream));

    std::cout<<"----------------finished initializer--------------- "<< std::endl;
}


void Yolo_Det::destroy_context(){
    bool cudart_ok = true;

    // release TensorRT 
    if(context){
        context->destroy();
        context = nullptr;
    }
    if(engine_det){
        engine_det->destroy();
        engine_det = nullptr;
    }
     if(engine_det == nullptr){
        runtime->destroy();  
    }

    // release cuda host memory
    if(stream){
        checkRuntime(cudaStreamDestroy(stream));}
    if(device_input_gpu){
        checkRuntime(cudaFree(device_input_gpu));}
    if(device_input_gpu){
        checkRuntime(cudaFree(device_output_gpu));}
    if(host_input_cpu){
        checkRuntime(cudaFreeHost(host_input_cpu));}
    if(host_output_cpu){
        checkRuntime(cudaFreeHost(host_output_cpu));}
}


void Yolo_Det::pre_process(cv::Mat& image){

    // wapaffine
    float scale_x = input_w / (float)image.cols;
    float scale_y = input_h / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_w + scale - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_h + scale - 1) * 0.5;
    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);  
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);  

    cv::Mat input_image(input_h, input_w, CV_8UC3);
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), 
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));  
    // cv::imwrite("input_image.jpg", input_image);

    // std::cout << "-----------input_image shape: " << input_image.cols << "*" 
    // << input_image.rows << "------------"<< std::endl;

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = host_input_cpu + image_area * 0;
    float* phost_g = host_input_cpu + image_area * 1;
    float* phost_r = host_input_cpu + image_area * 2;

    // rgbrgbrgb->rrrgggbbb
    for(int i = 0; i < image_area; ++i, pimage += 3)
    {
        *phost_r++ = pimage[0] / 255.0;
        *phost_g++ = pimage[1] / 255.0;
        *phost_b++ = pimage[2] / 255.0;
    }

    // host-->device
    checkRuntime(cudaMemcpyAsync(device_input_gpu, host_input_cpu, input_buffer_size,
                    cudaMemcpyHostToDevice, stream));
}


void Yolo_Det::inference(cv::Mat& img)
{
    assert(context != nullptr);
    // pre_process_time
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    pre_process(img);
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    float preprocess_time = std::chrono::duration<float, std::milli>(end_preprocess - start_preprocess).count();
   
    // std::cout << "-----------pre_process time: "<< preprocess_time<<" ms.------------"<< std::endl;
    
    // inference_time
    auto start_inference = std::chrono::high_resolution_clock::now();
    float* bindings[] = {device_input_gpu, device_output_gpu};
    
    context->enqueueV2((void**)bindings, stream, nullptr);
    auto end_inference = std::chrono::high_resolution_clock::now();
    float inference_time = std::chrono::duration<float, std::milli>(end_inference - start_inference).count();

    // std::cout << "-----------inference time : " << inference_time << " ms.------------" << std::endl;

    // post_process_time
    auto start_postprocess = std::chrono::high_resolution_clock::now();
    post_process(img);
    auto end_postprocess = std::chrono::high_resolution_clock::now();
    float postprocess_time = std::chrono::duration<float, std::milli>(end_postprocess - start_postprocess).count();

    // std::cout << "-----------post_process time: " << postprocess_time << " ms.-----------\n" << std::endl;
}


void Yolo_Det::post_process(cv::Mat& img){
    
    checkRuntime(cudaMemcpyAsync(host_output_cpu, device_output_gpu,
                    output_buffer_size, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    std::vector<std::vector<float>> bboxes;
    
    // decode
    for (int i = 0; i < output_numbox; i++){
        float* ptr = host_output_cpu + i * output_numprob;
        float objness = ptr[4];
        
        if(objness < conf_thresh)
            continue;

        float* pclass = ptr + 5;
        int label     = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob    = pclass[label];
        float confidence = prob * objness;

        if(confidence < conf_thresh)
            continue;

        float cx     = ptr[0];
        float cy     = ptr[1];
        float width  = ptr[2];
        float height = ptr[3];
        float left   = cx - width * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width * 0.5;
        float bottom = cy + height * 0.5;
        float image_base_left   = d2i[0] * left   + d2i[2];
        float image_base_right  = d2i[0] * right  + d2i[2];
        float image_base_top    = d2i[0] * top    + d2i[5];
        float image_base_bottom = d2i[0] * bottom + d2i[5];
        bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence});
    }
  
    // nms
    std::sort(bboxes.begin(), bboxes.end(), [](std::vector<float>& a, std::vector<float>& b){return a[5] > b[5];});
    std::vector<bool> remove_flags(bboxes.size());
    std::vector<std::vector<float>> box_result;
    box_result.reserve(bboxes.size());

    auto iou = [](const std::vector<float>& a, const std::vector<float>& b){
        float cross_left   = std::max(a[0], b[0]);
        float cross_top    = std::max(a[1], b[1]);
        float cross_right  = std::min(a[2], b[2]);
        float cross_bottom = std::min(a[3], b[3]);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) 
                         + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area / union_area;
    };

    for(int i = 0; i < bboxes.size(); ++i){
        if(remove_flags[i]) continue;

        auto& ibox = bboxes[i];
        box_result.emplace_back(ibox);
        for(int j = i + 1; j < bboxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = bboxes[j];
            if(ibox[4] == jbox[4]){
                // class matched
                if(iou(ibox, jbox) >= nms_thresh)
                    remove_flags[j] = true;
            }
        }
    }

    // draw_boxes
    for(int i = 0; i < box_result.size(); ++i){
        auto& ibox = box_result[i];
        float left = ibox[0];
        float top = ibox[1];
        float right = ibox[2];
        float bottom = ibox[3];
        int class_label = ibox[4];
        float confidence = ibox[5];
        
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(class_label);
        cv::rectangle(img, cv::Point(left, top), cv::Point(right, bottom), color, 3);

        auto name      = cocolabels[class_label];
        auto caption   = cv::format("%s %.2f", name, confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(img, cv::Point(left-3, top-33), cv::Point(left + text_width, top), color, -1);
        cv::putText(img, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
}