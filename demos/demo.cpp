/*
File: demo.cpp
Author: Dylan Baxter
Date: 3/13/23
Description: 
    This file contains the main functionality for
    a demo script demonstrating tensorrt inference with yolov8
*/

#include "yolov8.hpp"
#include "opencv2/opencv.hpp"

namespace ty = jetsontrt::yolov8;

int main(int argc, char** argv){
    // Set Cuda Device
    cudaSetDeviced(0);

    // Create Config
    Configuration config;
    config.onnx_path = "~/models/yolov8_shark.onnx";
    config.eng_path = "~/models/yolov8_shark.eng";
    config.workspace_size = 1U << 30;

    // Initialize OpenCV Capture and Write Object
    video_src = "~/test_videos/shark.mp4";
    video_dest = "~/test_videos/output/inference_test.mp4";
    cv::VideoCapture cap(video_src);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    float fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer(video_dest,
        cap.get(cv::CAP_PROP_FOURCC),
        fps,
        cv::Size(w, h));

    // Initialize Inference Object
    std::shared_ptr<ty::Yolov8> yolo(config);

    // Initialize Other Variables
    cv::Mat frame(h,w,CV_8UC3);
    std::vector<bbox> bbox_list;
    int c;

    // Video Capture Loop
    do{
        cap >> frame;
        if(frame.empty()){
            break;
        }

        // Perform Inference
        bbox_list = yolo->RunYOLOv8(frame);
        
        //Show frame
        cv::imshow("Frame", frame);
        c = static_cast<char>(waitKey(1));
        writer.write(frame);

    }while((c != ESCAPE_KEY));

    yolo->FreeBuffers();
    writer.close();

    return 0;
}