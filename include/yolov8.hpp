/*
 * File: yolov8.hpp
 * Author: Dylan Baxter
 * Date: 3/8/23
 * Description:
 *     This file contains declarations for functions and
 *     variables necessary for the functioning of YOLOv8
 *     in TensorRT.
 */

#ifndef YOLOV8_HPP_
#define YOLOV8_HPP_

// Includes
#include "opencv4/opencv2/opencv.hpp"
#include "generic.hpp"

// Namespace Declarations
namespace jetsontrt::yolov8 {

// Structure for bounding boxes
struct Bbox {
  int cls_id;
  std::string cls_name;
  float conf;
  int x;
  int y;
  int w;
  int h;
};

// YOLOv8 Inference Class
class Yolov8 : public Inferer {
 protected:
  Configuration config_;

 public:
  explicit Yolov8(const Configuration& config);
  void preprocess(const cv::Mat& im);
  std::vector<Bbox> postprocess();
  std::vector<Bbox> RunYOLOv8(const cv::Mat& im);
};

// Function to draw bounding boxes on the image
void drawBoxes(const cv::Mat& im, const std::vector<Bbox>& bbox_list);

}  // namespace jetsontrt::yolov8

#endif  // YOLOV8_HPP_