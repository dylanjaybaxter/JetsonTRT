/*
 * File: generic.hpp
 * Author: Dylan Baxter
 * Date: 3/8/23
 * Description:
 *     This file contains the declaration of functions and
 *     variables necessary for general TensorRT operations.
 */

#ifndef GENERIC_HPP_
#define GENERIC_HPP_

// Includes
// C++ Standard Libraries
#include <string>
#include <fstream>
#include <memory>
#include <vector>

// CUDA Libraries
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"

// Namespace Definitions
namespace nvi = nvinfer1;
namespace nvp = nvonnxparser;

// CUDA Error Checking
#define CUDA_CHECK_CALL(call) { gpuAssert((call), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    spdlog::error("CUDA ERROR: {} in {} at line {}", cudaGetErrorString(code), file, line);
    if (abort) {
      std::exit(code);
    }
  }
}

// Function Declarations
namespace jetsontrt {

struct Configuration {
  std::string onnx_path;
  std::string eng_path;
  std::size_t workspace_size;
};

class Inferer {
 protected:
  Configuration config_;
  int workspace_size_;
  int input_size_[2];
  std::vector<void*> in_buff_;
  std::vector<void*> out_buff_;
  std::vector<nvi::Dims32> input_dims_;
  std::vector<nvi::Dims32> output_dims_;
  std::shared_ptr<nvi::ICudaEngine> engine_;
  std::shared_ptr<nvi::IExecutionContext> context_;
  std::shared_ptr<nvp::IParser> parser_;
  std::shared_ptr<nvi::IRuntime> runtime_;
  std::shared_ptr<cudaStream_t> stream_;

 public:
  explicit Inferer(const Configuration& config);
  void BuildEngine(const std::string& onnx_path, const std::string& eng_path);
  void ReadEngine(const std::string& eng_path);
  void AllocateBuffers();
  void BuildContext();
  void Infer();
  void FreeBuffers();
};

}  // namespace jetsontrt

#endif  // GENERIC_HPP_
