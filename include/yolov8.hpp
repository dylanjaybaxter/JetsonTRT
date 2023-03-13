/*
File: yolov8.hpp
Author: Dylan Baxter
Date: 3/8/23
Description: 
    This file contains declaration for functions and
    variables necessary for the functioning of yolov8
    in tensorrt.
*/
// Includes
#include "opencv2/opencv2"
#include "generic.hpp"


// Constants
std::map<int, std::string> kNameMap = {
    {0, "shark"},
    {1, "boatdolphin"},
    {2, "person"},
    {3, "seal"}
}


// Functions

namespace jetsontrt::yolov8{
    class Yolov8 : public Inferer{
        protected:
        Configuration config_;


        public:
        void Yolov8(const Configuration config);
        void preprocess(cv::Mat im);
        cv::Mat postprocess(std::vector<void*> outBuff);
        std::vector<bboxes> RunYOLOv8(cv::Mat im);
    }
    
}