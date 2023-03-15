/*
File: generic.cpp
Author: Dylan Baxter
Date: 3/8/23
Description: 
    This file contains declaration of functions and
    variables necessary for general tensorrt operations 
*/
#ifndef GENERIC_CPP_
#define GENERIC_CPP_

// C++ libaries
#include <iostream>
#include <iterator>

// This Library
#include "generic.hpp"

// Image Processing
#include "opencv4/opencv2/opencv.hpp"

// Namespace Definitions
namespace nvi = nvinfer1;
namespace nvp = nvonnxparser;

//Logger Definition
class Logger : public nvi::ILogger {
    void log(Severity severity, const char* msg) noexcept override{
        if (true){
            std::cout << msg << "\n";
        }
    }
}logger_;

namespace jetsontrt {
    Inferer::Inferer(const Configuration config){
        // Check for engine file
        std::ifstream file(config.eng_path);
        if(!file.good())
            BuildEngine(config.onnx_path, config.eng_path);
        
        // Read in Engine
        ReadEngine(config.eng_path);

        // Generate Execution Context
        BuildContext();

        // Allocate Buffers for IO
        AllocateBuffers();

    }
    void Inferer::BuildEngine(std::string onnx_path, std::string eng_path){
        // Create Builder with Logger
        auto builder = std::shared_ptr<nvi::IBuilder>(
            nvi::createInferBuilder(logger_));
        // Create Network Definition
        uint32_t flag = 1U << static_cast<uint32_t>(
            nvi::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
        );
        auto network = std::shared_ptr<nvi::INetworkDefinition>(
            builder->createNetworkV2(flag)
        );

        // Create Parser
        parser_ = std::shared_ptr<nvp::IParser>(
            nvp::createParser(*network, logger_)
        );

        // Parse Model
        parser_->parseFromFile(onnx_path.c_str(), 
            static_cast<std::int32_t>(nvi::ILogger::Severity::kWARNING));
        for(int i=0; i < parser_->getNbErrors();i++){
            std::cout << parser_->getError(i)->desc() << "\n";
        }
        // Create Configuration
        auto config = std::shared_ptr<nvi::IBuilderConfig>(
            builder->createBuilderConfig()
        );
        
        if(builder->platformHasFastFp16()){
            std::cout << "Configured with fp16\n";
            config->setFlag(nvi::BuilderFlag::kFP16);
        }

        // Serialize Engine
        if(!network || !config || !parser_){
            std::cout << "Invalid Build\n";
        }
        auto serializedEngine = std::shared_ptr<nvi::IHostMemory>(
            builder->buildSerializedNetwork(*network, *config)
        );
        if (!serializedEngine){
            std::cout << "Serialization Failure!\n";
        }

        // Write Engine to File
        std::ofstream out_file(eng_path, std::ios::binary);
        out_file.write(
            reinterpret_cast<const char*>(serializedEngine->data()),
            serializedEngine->size()
        );
        out_file.close();
    }
    void Inferer::ReadEngine(std::string eng_path){
        // Read in the file contents
        std::ifstream in_file(eng_path, std::ios::binary);
        std::string buffer;
        if(!in_file){
            std::cout << "Could not read engine file: " << eng_path << "\n";
        }else{
            in_file >> std::noskipws;
            std::copy(std::istream_iterator<char>(in_file), std::istream_iterator<char>(), back_inserter(buffer));
        }

        // Initialize Plugins and Create Runtime
        initLibNvInferPlugins(&logger_, "");
        runtime_ = std::shared_ptr<nvi::IRuntime>(
            nvi::createInferRuntime(logger_)
        );

        // Deserialize Engine
        std::cout << "Deserialize Engine of size: " << buffer.size() << "\n";
        engine_ = std::shared_ptr<nvi::ICudaEngine>(
            runtime_->deserializeCudaEngine(buffer.data(), buffer.size())
        );
        if(engine_){
            std::cout <<"Deserialized Successfully!\n";
        }else{
            std::cout << "Bad Deserialization:(\n";
        }
        in_file.close();

        // Create Stream
        //stream_ = cudaStreamCreate(in_file);

        // Extract IO Properties
        //AllocateBuffers();

    }
    void Inferer::AllocateBuffers(){
        std::cout << engine_ << " is allocating Buffers...\n";
        std::cout << "With " <<  engine_->getNbBindings() << " bindings\n";
        // For each buffer in input and output
        // Allocate memory, name etc and store pointers
        int num_input = 0;
        int num_output = 0;
        for(int i=0; i < engine_->getNbBindings(); i++){
            // Get Binding Dimensions
             std::cout << "Ind: " << i << "\n";
            std::cout << "Name: " << engine_->getBindingName(i) << "\n";
            std::cout << "Desc: " << engine_->getBindingFormatDesc(i) << "\n";
            auto dims = engine_->getBindingDimensions(i);
            std::cout << "Num Dims: " << dims.nbDims << "\n";
            // Calculate Binding Size
            int bindingSize = 1;
            for(int j=0; j < dims.nbDims; j++){
                std::cout << "Dim " << j << " - ";
                std::cout << dims.d[j] << "\n";
                bindingSize *= dims.d[j];
                 
            }
            std::cout << "Allocating Buffer of size: " << bindingSize << "\n";
            // Allocate Memory
            void* data_ptr = nullptr;
            CUDA(cudaMalloc(&data_ptr, bindingSize*sizeof(_Float32)));
            // Add to Inputs/Outputs
            if(engine_->bindingIsInput(i)){
                num_input++;
                inBuff_.emplace_back(data_ptr);
                input_dims_.emplace_back(dims);
            }else{
                num_output++;
                outBuff_.emplace_back(data_ptr);
                output_dims_.emplace_back(dims);
            }
        }

    }
    void Inferer::BuildContext(){
       // Create Context with Engine
        if (engine_){
            context_.reset(engine_->createExecutionContext());
        }
        std::cout << "Context Built\n";
    }
    void Inferer::Infer(){
        // Concatenate the bindings
        std::vector<void*> bnd;
        bnd.reserve(inBuff_.size()+outBuff_.size());
        bnd.insert(bnd.end(), inBuff_.begin(), inBuff_.end());
        bnd.insert(bnd.end(), outBuff_.begin(), outBuff_.end());
        // Check Validity
        for (int i=0; i < bnd.size(); i++){
            assert(bnd[i]);
        }
        // Run Inference
        bool stat = context_->executeV2(bnd.data());
        if(!stat){
            std::cout << "Bad Enqueue\n";
        }
    }
    void Inferer::FreeBuffers(){
        // Free All  Allocated Memory
        for(int i = 0; i < inBuff_.size(); i++){
            cudaFree(inBuff_[i]);
        }
        for(int i = 0; i < outBuff_.size(); i++){
            cudaFree(outBuff_[i]);
        }

    }
    
}
#endif //GENERIC_CPP_


