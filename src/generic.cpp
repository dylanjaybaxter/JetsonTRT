/*
 * File: generic.cpp
 * Author: Dylan Baxter
 * Date: 3/8/23
 * Description:
 *     This file contains the declaration of functions and
 *     variables necessary for general TensorRT operations.
 */

#ifndef GENERIC_CPP_
#define GENERIC_CPP_

// C++ Libraries
#include <iostream>
#include <iterator>
#include <fstream>
#include <memory>
#include <vector>
#include <spdlog/spdlog.h>

// This Library
#include "generic.hpp"

// Image Processing
#include "opencv4/opencv2/opencv.hpp"

// Namespace Definitions
namespace nvi = nvinfer1;
namespace nvp = nvonnxparser;

// Logger Definition
class Logger : public nvi::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity != Severity::kINFO) {
      spdlog::info(msg); // Use spdlog for info-level logging
    }
  }
} logger_;

namespace jetsontrt {

// Inferer constructor
Inferer::Inferer(const Configuration& config) {
  std::ifstream file(config.eng_path);
  if (!file.good()) {
    BuildEngine(config.onnx_path, config.eng_path);
  }

  ReadEngine(config.eng_path);

  BuildContext();

  AllocateBuffers();
}

// Build Engine
void Inferer::BuildEngine(const std::string& onnx_path, const std::string& eng_path) {
  auto builder = std::shared_ptr<nvi::IBuilder>(nvi::createInferBuilder(logger_));

  uint32_t flag = 1U << static_cast<uint32_t>(nvi::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::shared_ptr<nvi::INetworkDefinition>(builder->createNetworkV2(flag));

  parser_ = std::shared_ptr<nvp::IParser>(nvp::createParser(*network, logger_));

  parser_->parseFromFile(onnx_path.c_str(), static_cast<int32_t>(nvi::ILogger::Severity::kWARNING));
  for (int i = 0; i < parser_->getNbErrors(); i++) {
    spdlog::warn(parser_->getError(i)->desc()); 
  }

  auto config = std::shared_ptr<nvi::IBuilderConfig>(builder->createBuilderConfig());

  if (builder->platformHasFastFp16()) {
    spdlog::info("Configured with fp16");
    config->setFlag(nvi::BuilderFlag::kFP16);
  }

  if (!network || !config || !parser_) {
    spdlog::error("Invalid Build");
    return;
  }

  auto serialized_engine = std::shared_ptr<nvi::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));

  if (!serialized_engine) {
    spdlog::error("Serialization Failure!");
    return;
  }

  std::ofstream out_file(eng_path, std::ios::binary);
  out_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
  out_file.close();
}

// Read Engine
void Inferer::ReadEngine(const std::string& eng_path) {
  std::ifstream in_file(eng_path, std::ios::binary);
  std::string buffer;
  if (!in_file) {
    spdlog::error("Could not read engine file: {}", eng_path);
    return;
  }

  in_file >> std::noskipws;
  std::copy(std::istream_iterator<char>(in_file), std::istream_iterator<char>(), std::back_inserter(buffer));

  initLibNvInferPlugins(&logger_, "");
  runtime_ = std::shared_ptr<nvi::IRuntime>(nvi::createInferRuntime(logger_));

  spdlog::info("Deserialize Engine of size: {}", buffer.size());
  engine_ = std::shared_ptr<nvi::ICudaEngine>(
      runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));

  if (engine_) {
    spdlog::info("Deserialized Successfully!");
  } else {
    spdlog::error("Bad Deserialization :(");
  }

  in_file.close();
}

// Allocate Buffers
void Inferer::AllocateBuffers() {
  spdlog::info("{} is allocating Buffers...", engine_);
  spdlog::info("With {} bindings", engine_->getNbBindings());

  int num_input = 0;
  int num_output = 0;
  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    spdlog::info("Ind: {}", i);
    spdlog::info("Name: {}", engine_->getBindingName(i));
    auto dims = engine_->getBindingDimensions(i);
    spdlog::info("Num Dims: {}", dims.nbDims);

    int binding_size = 1;
    for (int j = 0; j < dims.nbDims; ++j) {
      spdlog::info("Dim {} - {}", j, dims.d[j]);
      binding_size *= dims.d[j];
    }

    spdlog::info("Allocating Buffer of size: {}", binding_size);

    void* data_ptr = nullptr;
    CUDA(cudaMalloc(&data_ptr, binding_size * sizeof(_Float32)));

    if (engine_->bindingIsInput(i)) {
      ++num_input;
      in_buff_.emplace_back(data_ptr);
      input_dims_.emplace_back(dims);
    } else {
      ++num_output;
      out_buff_.emplace_back(data_ptr);
      output_dims_.emplace_back(dims);
    }
  }
}

// Build Context
void Inferer::BuildContext() {
  if (engine_) {
    context_.reset(engine_->createExecutionContext());
  }
  spdlog::info("Context Built");
}

// Inference
void Inferer::Infer() {
  std::vector<void*> bindings;
  bindings.reserve(in_buff_.size() + out_buff_.size());
  bindings.insert(bindings.end(), in_buff_.begin(), in_buff_.end());
  bindings.insert(bindings.end(), out_buff_.begin(), out_buff_.end());

  for (const auto* binding : bindings) {
    assert(binding);
  }

  bool status = context_->executeV2(bindings.data());
  if (!status) {
    spdlog::error("Bad Enqueue");
  }
}

// Free Buffers
void Inferer::FreeBuffers() {
  for (auto* buffer : in_buff_) {
    cudaFree(buffer);
  }

  for (auto* buffer : out_buff_) {
    cudaFree(buffer);
  }
}

}  // namespace jetsontrt

#endif  // GENERIC_CPP_
