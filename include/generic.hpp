/*
File: generic.cpp
Author: Dylan Baxter
Date: 3/8/23
Description: 
    This file contains definition of functions and
    variables necessary for general tensorrt operations 
*/
#ifndef GENERIC_HPP_
#define GENERIC_HPP_

// Includes
// C++ Std libraries
#include <string>
#include <fstream>
#include <iostream>
#include <memory>

// Cuda Libraries
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"

// Namespace
namespace nvi = nvinfer1;
namespace nvp = nvonnxparser;

// Constants

// CUDA Error Checking
#define CUDA(thing) { gpuAssert((thing), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr, "CUDA ERROR: %s %s %d\n", cudaGetErrorString(code),
            file, line);
        if (abort){
            exit(code);
        }
    }
}

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
    std::shared_ptr<nvp::IParser> parser_;
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
};
}
#endif //GENERIC_HPP