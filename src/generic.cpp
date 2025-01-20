/*
 * File: generic.cpp
 * Author: Dylan Baxter
 * Date: 3/8/23
 * Description:
 *     This file contains the implementation of functions and
 *     variables necessary for general TensorRT operations.
 */

#include "generic.hpp"
#include <fstream>
#include <filesystem>
#include <spdlog/spdlog.h>
#include "opencv4/opencv2/opencv.hpp"

namespace jetsontrt {

Inferer::Inferer(const Configuration& config) : config_(config) {
  std::ifstream file(config.eng_path);
  if (!file.good()) {
    buildEngine(config.onnx_path, config.eng_path);
  }

  // Checking for Cached Engine (build takes a long time)
  if(std::filesystem::exists(config.eng_path)){
    readEngineFromFile(config.eng_path);
  } else {
    readEngineFromOnnx(config.onnx_path, config.eng_path);
  }
  buildContext();
  allocateBuffers();
}

Inferer::~Inferer() {
  // Free buffers on destruct
  for (auto* buffer : in_buff_) {
    cudaFree(buffer);
  }

  for (auto* buffer : out_buff_) {
    cudaFree(buffer);
  }
}

bool Inferer::readEngineFromOnnx(const std::string& onnx_path, const std::string& eng_path) {
  auto builder = std::shared_ptr<nvi::IBuilder>(nvi::createInferBuilder(logger_));
  uint32_t flag = 1U << static_cast<uint32_t>(nvi::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::shared_ptr<nvi::INetworkDefinition>(builder->createNetworkV2(flag));
  parser_ = std::shared_ptr<nvp::IParser>(nvp::createParser(*network, logger_));

  if (!std::filesystem::exists(onnx_path)) {
    spdlog::error("ONNX Path Not Found: {}", onnx_path);
    return false;
  }

  parser_->parseFromFile(onnx_path.c_str(), static_cast<int32_t>(nvi::ILogger::Severity::kWARNING));
  if (parser_->getNbErrors() > 0) {
    std::ostringstream oss;
    for (int i = 0; i < parser_->getNbErrors(); i++) {
      oss << "ONNX Parsing Error " << i << ": " << parser_->getError(i)->desc() << "\n";
    }
    spdlog::warn(oss.str());
  }

  auto config = std::shared_ptr<nvi::IBuilderConfig>(builder->createBuilderConfig());
  if (builder->platformHasFastFp16()) {
    spdlog::info("Configured with FP16");
    config->setFlag(nvi::BuilderFlag::kFP16);
  }
  if (!network || !config || !parser_) {
    spdlog::error("Invalid Build");
    return false;
  }

  auto serialized_engine = std::shared_ptr<nvi::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  if (!serialized_engine) {
    spdlog::error("Serialization Failure");
    return false;
  }

  buildEngine(serialized_engine);
  if (!engine_) {
    spdlog::error("Engine not built");
    return false;
  }

  std::ofstream out_file(eng_path, std::ios::binary);
  out_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
  return true;
}

void Inferer::allocateBuffers() {
  spdlog::info("{} is allocating Buffers...", engine_);
  spdlog::info("With {} bindings", engine_->getNbBindings());

  int num_input = 0;
  int num_output = 0;
  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    auto dims = engine_->getBindingDimensions(i);
    int binding_size = 1;
    for (int j = 0; j < dims.nbDims; ++j) {
      binding_size *= dims.d[j];
    }
    void* data_ptr = nullptr;
    CUDA(cudaMalloc(&data_ptr, binding_size * sizeof(float)));

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

void Inferer::buildContext() {
  if (engine_) {
    context_.reset(engine_->createExecutionContext());
  }
  spdlog::info("Context Built");
}

bool Inferer::infer() {
  std::vector<void*> bindings;
  bindings.reserve(in_buff_.size() + out_buff_.size());
  bindings.insert(bindings.end(), in_buff_.begin(), in_buff_.end());
  bindings.insert(bindings.end(), out_buff_.begin(), out_buff_.end());

  bool status = context_->executeV2(bindings.data());
  if (!status) {
    spdlog::error("Bad Enqueue");
    return false;
  }
  return true;
}

}  // namespace jetsontrt
