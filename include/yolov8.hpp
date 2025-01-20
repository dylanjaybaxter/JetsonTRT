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

#include "opencv4/opencv2/opencv.hpp"
#include "generic.hpp"

const std::map<int, std::string> kNameMap = {
    {0, "shark"},
    {1, "boatdolphin"},
    {2, "person"},
    {3, "seal"}
};

namespace jetsontrt::yolov8 {

/**
 * @brief Structure representing a bounding box.
 */
struct Bbox {
  int cls_id;            /// Class ID of the detected object
  std::string cls_name;  /// Name of the detected class
  float conf;            /// Confidence score of the detection
  int x;                 /// X-coordinate of the top-left corner
  int y;                 /// Y-coordinate of the top-left corner
  int w;                 /// Width of the bounding box
  int h;                 /// Height of the bounding box
};

class Yolov8 : public Inferer {
 protected:
  Configuration config_;

 public:
  /**
   * @brief Constructs a Yolov8 object with the given configuration.
   * @param config The configuration specifying paths and workspace size.
   */
  explicit Yolov8(const Configuration& config);

  /**
   * @brief Runs inference on the given image and returns detected bounding boxes.
   * @param im Input image in OpenCV Mat format.
   * @return A vector of detected bounding boxes.
   */
  std::vector<Bbox> runYolov8(const cv::Mat& im);

 private:
  /**
   * @brief Preprocesses the input image before inference.
   * @param im Input image in OpenCV Mat format.
   */
  void preprocess(const cv::Mat& im);

  /**
   * @brief Processes inference output and extracts bounding boxes.
   * @return A vector of detected bounding boxes.
   */
  std::vector<Bbox> postprocess();
};

/**
 * @brief Draws bounding boxes on the given image.
 * @param im Input image.
 * @param bbox_list List of bounding boxes to be drawn.
 */
void drawBoxes(const cv::Mat& im, const std::vector<Bbox>& bbox_list);

}  // namespace jetsontrt::yolov8

#endif  // YOLOV8_HPP_
