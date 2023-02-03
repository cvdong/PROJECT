/*
 * @Version: v1.0
 * @Author: 東DONG
 * @Mail: cv_yang@126.com
 * @Date: 2023-01-16 09:08:20
 * @LastEditTime: 2023-01-31 15:42:57
 * @FilePath: /YOLO_TRT/src/main.cpp
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

#include "include/yolo.h"

namespace YOLO{

    namespace log{

        #define INFO(...)  YOLO::log::__printf(__FILE__, __LINE__, __VA_ARGS__)

        void __printf(const char* file, int line, const char* fmt, ...){

            va_list vl;
            va_start(vl, fmt);

            printf("\e[32m[%s:%d]:\e[0m ", file, line);
            vprintf(fmt, vl);
            printf("\n");
        }
    };


    int run_image(){

        const std::string img_path = "../workspace/bus.jpg";
        cv::Mat image = cv::imread(img_path);
        
        if(image.empty()){
            std::cout<<"Input image path wrong!!"<<std::endl;
            return -1;
        }

        const std::string model_path = "../workspace/yolov8s_fp16.engine";
        Yolo_Det* yolo_instance = new Yolo_Det(model_path);
        
        auto start_total = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < 1000; i++){

            yolo_instance->inference(image);} 
        auto end_total = std::chrono::high_resolution_clock::now();
        
        float total_time = std::chrono::duration<float, std::milli>(end_total - start_total).count();
    
        INFO("total time:%f ms.", total_time/1000);

        if(yolo_instance)
        {
            delete yolo_instance;
            yolo_instance = nullptr;
        }

        return 0;
    }


    int run_video(){

        const std::string video_path = "../workspace/vtest.avi";
        const std::string model_path = "../workspace/yolov8s_fp16.engine";
        Yolo_Det* yolo_instance = new Yolo_Det(model_path);

        cv::Mat frame;
        cv::VideoCapture cap(video_path);

        if(!cap.isOpened()){
            INFO("Could not open the input video :");
            return -1;
        }

        for(;;){
            cap >> frame;
            if(frame.empty()) break;

            auto start_total = std::chrono::high_resolution_clock::now();
            yolo_instance->inference(frame);
            auto end_total = std::chrono::high_resolution_clock::now();
        
            float total_time = std::chrono::duration<float, std::milli>(end_total - start_total).count();

            std::stringstream fpss;
            fpss << "FPS:" << float(1000.0f / total_time);

            cv::putText(frame, fpss.str(), cv::Point(0, 25), 0, 1, cv::Scalar::all(0), 2, 16);

            INFO("total time:%f ms fps: %f. \n", total_time, float(1000.0f / total_time));

            cv::imshow("test", frame);
            if(cv::waitKey(1) == 27) break;
        }

        cap.release();
        cv::destroyAllWindows();

        return 0;
    }

}


int main(int argc, char** argv){

    // YOLO::run_image();
    YOLO::run_video();
}