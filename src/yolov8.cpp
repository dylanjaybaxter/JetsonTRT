/*
 * File: yolov8.cpp
 * Author: Dylan Baxter
 * Date: 3/8/23
 * Description:
 *     This file contains definitions for functions and
 *     variables necessary for the functioning of YOLOv8
 *     in TensorRT.
 */

#ifndef YOLOV8_CPP_
#define YOLOV8_CPP_

#include "yolov8.hpp"
#include <spdlog/spdlog.h>

// Constants
const std::map<int, std::string> kNameMap = {
    {0, "shark"},
    {1, "boatdolphin"},
    {2, "person"},
    {3, "seal"}
};

namespace jetsontrt::yolov8 {

Yolov8::Yolov8(const Configuration& config) : Inferer(config) {
  // Explicitly call the generic constructor
}

std::vector<bbox> Yolov8::RunYOLOv8(const cv::Mat& im) {
  // Preprocess image
  preprocess(im);

  // Run inference
  Infer();

  // Post process output
  return postprocess();
}

void Yolov8::preprocess(const cv::Mat& im) {
  // Get network input size
  int net_h = input_dims_.at(0).d[2];
  int net_w = input_dims_.at(0).d[3];

  // Read image size
  input_size_[0] = im.cols;
  input_size_[1] = im.rows;

  // Preallocate Mats
  cv::Mat resize_mat(net_h, net_w, CV_8UC3);
  cv::Mat color_mat(net_h, net_w, CV_8UC3);
  cv::Mat norm_mat(net_h, net_w, CV_32FC3);

  // Resize image
  cv::resize(im, resize_mat, resize_mat.size(), 0, 0, cv::INTER_LINEAR);

  // Convert image from BGR to RGB
  cv::cvtColor(resize_mat, color_mat, cv::COLOR_BGR2RGB);

  // Split the image into channels and normalize
  color_mat.convertTo(norm_mat, CV_32FC3, 1.0 / 255.0);

  // Save each channel in series to the input buffer
  std::vector<cv::Mat> channels(3);
  cv::split(norm_mat, channels);

  assert(inBuff_.at(0));
  int mat_size = net_h * net_w;
  for (int i = 0; i < channels.size(); i++) {
    CUDA(cudaMemcpy(reinterpret_cast<_Float32*>(inBuff_.at(0)) + (mat_size * i),
                    channels[i].data, mat_size * sizeof(_Float32), cudaMemcpyHostToDevice));
  }
}

std::vector<bbox> Yolov8::postprocess() {
  // Create output storage
  int32_t nbDet;
  int bbox_size = output_dims_[1].d[1] * output_dims_[1].d[2];
  float32_t bbox_data[bbox_size];
  int other_size = output_dims_[1].d[1];
  float32_t scores[other_size];
  float32_t labels[other_size];

  // Copy output bindings to buffers
  CUDA(cudaMemcpy(reinterpret_cast<int32_t*>(&nbDet),
                  reinterpret_cast<int32_t*>(outBuff_.at(0)),
                  sizeof(int32_t), cudaMemcpyDeviceToHost));

  CUDA(cudaMemcpy(reinterpret_cast<_Float32*>(&bbox_data[0]),
                  reinterpret_cast<_Float32*>(outBuff_.at(1)),
                  bbox_size * sizeof(_Float32), cudaMemcpyDeviceToHost));

  CUDA(cudaMemcpy(reinterpret_cast<_Float32*>(&scores[0]),
                  reinterpret_cast<_Float32*>(outBuff_.at(2)),
                  other_size * sizeof(_Float32), cudaMemcpyDeviceToHost));

  CUDA(cudaMemcpy(reinterpret_cast<_Float32*>(&labels[0]),
                  reinterpret_cast<_Float32*>(outBuff_.at(3)),
                  other_size * sizeof(_Float32), cudaMemcpyDeviceToHost));

  // Process output blobs into a list of bboxes
  std::vector<bbox> bbox_list;

  // Bboxes are scaled by input size
  float fx = static_cast<float>(input_size_[0]) / static_cast<float>(input_dims_[0].d[2]);
  float fy = static_cast<float>(input_size_[1]) / static_cast<float>(input_dims_[0].d[3]);

  // Now extract bboxes in format (x1, y1, x2, y2)
  for (int i = 0; i < nbDet; i++) {
    bbox box;

    box.x = bbox_data[i * 4] * fx;
    box.y = bbox_data[(i * 4) + 1] * fy;
    box.w = (bbox_data[(i * 4) + 2] - bbox_data[i * 4]) * fx;
    box.h = (bbox_data[(i * 4) + 3] - bbox_data[(i * 4) + 1]) * fy;
    box.conf = scores[i];
    box.cls_id = labels[i];

    if (kNameMap.find(box.cls_id) != kNameMap.end()) {
      box.cls_name = kNameMap.at(box.cls_id);
    } else {
      box.cls_name = "Unimplemented";
    }

    bbox_list.emplace_back(box);
  }

  return bbox_list;
}

void drawBoxes(const cv::Mat& im, const std::vector<bbox>& bbox_list) {
  for (const auto& box : bbox_list) {
    int label = box.cls_id;
    const std::string& label_name = box.cls_name;
    int x = box.x;
    int y = box.y;
    int w = box.w;
    int h = box.h;

    cv::Rect roi(x, y, w, h);
    auto color = cv::Scalar(0, 255, 0);
    int thickness = 1;

    cv::rectangle(im, roi, color, thickness);
    cv::putText(im, label_name, cv::Point(x, y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 2);
  }
}

}  // namespace jetsontrt::yolov8

#endif  // YOLOV8_CPP_
