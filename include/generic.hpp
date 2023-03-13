/*
File: generic.cpp
Author: Dylan Baxter
Date: 3/8/23
Description: 
    This file contains definition of functions and
    variables necessary for general tensorrt operations 
*/
// Includes
// C++ Std libraries
#include <string>
#include <fstream>
#include <iostream>

// Cuda Libraries
#include "cuda_runtime.h"
#include "cuda.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"

// Jetpack Libraries
#include "jetson-utils/logging.h"
#include "jetson-utils/cudaMappedMemory.h"
#include "jetson-utils/imageIO.h"
#include "jetson-utils/imageIO.h"

// Namespace
namespace nvi = nvinfer1;
namespace nvp = nvonnxparser;

// Constants


// Functions
namespace jetsontrt {

struct Configuration{
    std::string onnx_path;
    std::string eng_path;
    std::size_t workspace_size;
};    

class Inferer{
    protected:
    Configuration config_;
    int workspace_size_;
    int input_size_[2];
    std::vector<void*> inBuff_;
    std::vector<void*> outBuff_;
    std::vector<nvi::Dims32> input_dims_;
    std::vector<nvi::Dims32> output_dims_;
    std::shared_ptr<nvi::ICudaEngine> engine_;
    std::shared_ptr<nvi::IExecutionContext> context_;
    std::shared_ptr<nvi::IParser> parser_;
    std::shared_ptr<nvi::IRuntime> runtime_;
    std::shared_ptr<cudaStream_t> stream_;

    public:
    Inferer(const Configuration config);
    void BuildEngine(std::string onnx_path, std::string eng_path);
    void ReadEngine(std::string eng_path);
    void AllocateBuffers();
    void BuildContext();
    void Infer();
    void FreeBuffers();
}


}