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

const int ESCAPE_KEY = 27;

namespace ty = jetsontrt::yolov8;
namespace jt = jetsontrt;

int main(int argc, char** argv){
    // Set Cuda Device
    //cudaSetDevice(0);

    // Create Config
    jt::Configuration config;
    config.onnx_path = "/home/jetson/models/shark_nms.onnx";
    config.eng_path = "/home/jetson/models/shark_nms.eng";
    config.workspace_size = 1 << 30;

    // Initialize OpenCV Capture and Write Object
    std::string video_src = "/home/jetson/test_videos/sharks.mp4";
    std::string video_dest = "/home/jetson/test_videos/output/inference_test.mp4";
    cv::VideoCapture cap(video_src);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    float fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer(video_dest,
        cap.get(cv::CAP_PROP_FOURCC),
        fps,
        cv::Size(w, h));

    // Initialize Inference Object
    auto yolo = 
        std::make_shared<ty::Yolov8>(ty::Yolov8(config));

    // Initialize Other Variables
    cv::Mat frame(h,w,CV_8UC3);
    std::vector<ty::bbox> bbox_list;
    char c = 0;

    // Video Capture Loop
    std::cout << "Starting Inference\n";
    do{
        cap >> frame;
        if(frame.empty()){
            break;
        }
        // Perform Inference
        bbox_list = yolo->RunYOLOv8(frame);

        //Draw Boxes
        ty::drawBoxes(frame, bbox_list);
        
        //Show frame
        writer.write(frame);
        cv::imshow("Frame", frame);
        c = static_cast<char>(cv::waitKey(1));
    }while((c != ESCAPE_KEY));

    yolo->FreeBuffers();
    writer.release();

    std::cout << "Done!\n";

    return 0;
}