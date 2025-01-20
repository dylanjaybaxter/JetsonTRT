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

#include <string>
#include <fstream>
#include <memory>
#include <vector>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"

namespace nvi = nvinfer1;
namespace nvp = nvonnxparser;

#define CUDA_CHECK_CALL(call) { gpuAssert((call), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    spdlog::error("CUDA ERROR: {} in {} at line {}", cudaGetErrorString(code), file, line);
    if (abort) {
      std::exit(code);
    }
  }
}

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
  /**
   * @brief Constructs an Inferer object with the given configuration.
   * @param config The configuration specifying paths and workspace size.
   */
  explicit Inferer(const Configuration& config);

  /**
   * @brief Runs inference on the loaded model.
   * @return True if inference is successful, false otherwise.
   */
  bool infer();

  ~Inferer();

  /**
   * @brief Reads an engine from an ONNX file and serializes it.
   * @param onnx_path Path to the ONNX model file.
   * @param eng_path Path to save the serialized engine.
   * @return True if successful, false otherwise.
   */
  bool readEngineFromOnnx(const std::string& onnx_path, const std::string& eng_path);

  /**
   * @brief Reads an engine from a serialized file.
   * @param eng_path Path to the serialized engine file.
   * @return True if successful, false otherwise.
   */
  bool readEngineFromFile(const std::string& eng_path);

  /**
   * @brief Writes the current engine to a file.
   * @param eng_path Path where the engine should be saved.
   * @return True if successful, false otherwise.
   */
  bool writeEngine(const std::string& eng_path);

  /**
   * @brief Builds an engine from a serialized engine buffer.
   * @param serialized_engine The buffer containing the serialized engine.
   * @return True if successful, false otherwise.
   */
  bool buildEngine(const std::vector<char>& serialized_engine);

  /**
   * @brief Allocates memory buffers for inference execution.
   */
  void allocateBuffers();

  /**
   * @brief Creates an execution context for inference.
   */
  void buildContext();
};

}  // namespace jetsontrt

#endif  // GENERIC_HPP_