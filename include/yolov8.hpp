/*
File: yolov8.hpp
Author: Dylan Baxter
Date: 3/8/23
Description: 
    This file contains declaration for functions and
    variables necessary for the functioning of yolov8
    in tensorrt.
*/
#ifndef YOLOV8_HPP_
#define YOLOV8_HPP_

// Includes
#include "opencv4/opencv2/opencv.hpp"
#include "generic.hpp"

// Functions
namespace jetsontrt::yolov8{
    struct bbox{
        int cls_id;
        std::string cls_name;
        float conf;
        int x;
        int y;
        int w;
        int h;
    };

    class Yolov8 : public Inferer{
        protected:
        Configuration config_;


        public:
        Yolov8(const Configuration config);
        void preprocess(const cv::Mat &im);
        std::vector<bbox> postprocess();
        std::vector<bbox> RunYOLOv8(const cv::Mat &im);
    };

    void drawBoxes(const cv::Mat &im,const std::vector<bbox> &bbox_list);
    
}
#endif //YOLOV8_HPP_