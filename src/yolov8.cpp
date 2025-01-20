/*
 * File: yolov8.cpp
 * Author: Dylan Baxter
 * Date: 3/8/23
 * Description:
 *     This file contains definitions for functions and
 *     variables necessary for the functioning of YOLOv8
 *     in TensorRT.
 */

#include "yolov8.hpp"
#include <spdlog/spdlog.h>

namespace jetsontrt::yolov8 {

Yolov8::Yolov8(const Configuration& config) : Inferer(config) {}

std::vector<Bbox> Yolov8::runYolov8(const cv::Mat& im) {
  // Preprocess image
  preprocess(im);

  // Run inference
  bool success = infer();

  // Post-process output if inference was successful
  return success ? postprocess() : std::vector<Bbox>();
}

void Yolov8::preprocess(const cv::Mat& im) {
  // Get network input size
  int net_h = input_dims_.at(0).d[2];
  int net_w = input_dims_.at(0).d[3];

  // Read image size
  input_size_[0] = im.cols;
  input_size_[1] = im.rows;

  // Preallocate Mats
  cv::Mat resize_mat, color_mat, norm_mat;
  cv::resize(im, resize_mat, cv::Size(net_w, net_h), 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(resize_mat, color_mat, cv::COLOR_BGR2RGB);
  color_mat.convertTo(norm_mat, CV_32FC3, 1.0 / 255.0);

  // Split channels and copy to input buffer
  std::vector<cv::Mat> channels(3);
  cv::split(norm_mat, channels);

  int mat_size = net_h * net_w;
  for (size_t i = 0; i < channels.size(); ++i) {
    CUDA(cudaMemcpy(reinterpret_cast<float*>(in_buff_.at(0)) + (mat_size * i),
                    channels[i].data, mat_size * sizeof(float), cudaMemcpyHostToDevice));
  }
}

std::vector<Bbox> Yolov8::postprocess() {
  int32_t num_detections;
  int bbox_size = output_dims_[1].d[1] * output_dims_[1].d[2];
  std::vector<float> bbox_data(bbox_size);
  std::vector<float> scores(output_dims_[1].d[1]);
  std::vector<float> labels(output_dims_[1].d[1]);

  // Copy output bindings to buffers
  CUDA(cudaMemcpy(&num_detections, out_buff_.at(0), sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(bbox_data.data(), out_buff_.at(1), bbox_size * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(scores.data(), out_buff_.at(2), scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(labels.data(), out_buff_.at(3), labels.size() * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<Bbox> bbox_list;
  float fx = static_cast<float>(input_size_[0]) / input_dims_[0].d[2];
  float fy = static_cast<float>(input_size_[1]) / input_dims_[0].d[3];

  for (int i = 0; i < num_detections; ++i) {
    Bbox box;
    box.x = bbox_data[i * 4] * fx;
    box.y = bbox_data[i * 4 + 1] * fy;
    box.w = (bbox_data[i * 4 + 2] - bbox_data[i * 4]) * fx;
    box.h = (bbox_data[i * 4 + 3] - bbox_data[i * 4 + 1]) * fy;
    box.conf = scores[i];
    box.cls_id = static_cast<int>(labels[i]);
    box.cls_name = kNameMap.count(box.cls_id) ? kNameMap.at(box.cls_id) : "Unknown";
    bbox_list.emplace_back(box);
  }
  return bbox_list;
}

void drawBoxes(cv::Mat& im, const std::vector<Bbox>& bbox_list) {
  for (const auto& box : bbox_list) {
    cv::Rect roi(box.x, box.y, box.w, box.h);
    cv::rectangle(im, roi, cv::Scalar(0, 255, 0), 2);
    cv::putText(im, box.cls_name, cv::Point(box.x, box.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
  }
}

}  // namespace jetsontrt::yolov8
