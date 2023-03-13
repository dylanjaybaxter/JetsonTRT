/*
File: generic.cpp
Author: Dylan Baxter
Date: 3/8/23
Description: 
    This file contains declaration of functions and
    variables necessary for general tensorrt operations 
*/

// C++ libaries
#include <ifstream>
#include <iostream>

// This Library
#include "generic.hpp"

// Image Processing
#include "opencv4/opencv2/opencv.hpp"

// Namespace Definitions
namespace nvi = nvinfer;
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

        // Allocate Buffers for IO
        AllocateBuffers();

        // Generate Execution Context
        BuildContext();

    }
    void Inferer::BuildEngine(std::string onnx_path, std::string eng_path){
        // Create Builder with Logger
        auto builder = std::shared_ptr<nvi::IBuilder>(
            logger_);
        // Create Network Definition
        uint32_t flag = 1U << static_cast<uint32_t>(
            nvi::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
        );
        auto network = std::shared_ptr<nvi::INetworkDefinition>(
            builder->createNetworkv2(flag)
        );

        // Create Parser
        parser_ = std::shared_ptr<nvp::IParser>(
            nvp::createParser(*network, logger)
        );

        // Parse Model
        parser_->parseFromFile(onnx_path.c_str(), 
            static_cast<std::int32_t>(nvi::ILogger::Severity::kWARNING));
        for(int i=0; i < parser_->getNbErrors();i++){
            std::cout << parser_getError(i)->desc() << "\n";
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
            builder->buildSerializedNetwork(*network, *config);
        );
        if (!serializedEngine){
            std::cout << "Serialization Failure!\n";
        }

        // Write Engine to File
        std::ofstream out_file(eng_path, std::os::binary);
        out_file.write(
            reinterpret_cast<const char*>(serializedEngine->data()),
            serializedEngine->size()
        );
        out_file.close();
    }
    void Inferer::ReadEngine(std::string eng_path){
        // Read in the file contents
        std::ifstream in_file(eng_path, std::os::binary);
        std::string buffer;
        if(!in_file){
            std::cout << "Could not read engine file: " << eng_path << "\n";
        }else{
            in_file >> std::noskipws;
            copy(std::istream_iterator<char>(in_file), std::istream_iterator<char>(), back_insterter(buffer));
        }

        // Initialize Plugins and Create Runtime
        initLibNvInferPlugins(&logger, "");
        runtime_ = std::shared_ptr<nvi::IRuntime>(
            nvi::createInferRuntime(logger)
        );

        // Deserialize Engine
        engine_ = std::shared_ptr<ICudaEngine>(
            runtime_->deserializeCudaEngine(buffer.data(), buffer.size())
        );
        in_file.close();

        // Create Context with Engine
        if (engine_){
            context_.reset(engine_->createExecutionContext());
        }
        // Create Stream
        stream_ = cudaStreamCreate(stream);

        // Extract IO Properties
        AllocateBuffers();

    }
    void Inferer::AllocateBuffers(){
        // For each buffer in input and output
        // Allocate memory, name etc and store pointers
        int num_input = 0;
        int num_output = 0;
        for(int i=0; i < engine_->getNbBindings(); i++){
            // Get Binding Dimensions
            nvi::Dims32 dims = engine_->getBindingDimensions(i);
            // Calculate Binding Size
            int bindingSize = 1;
            for(int j; j < dims.nbDims; j++){
                bindingSize *= dims.d[j];
            }
            // Allocate Memory
            void* data_ptr = nullptr;
            CUDA(cudaMalloc(&data_ptr, bindingSize));
            // Add to Inputs/Outputs
            if(engine_->bindingIsInput(i)){
                num_input++;
                inBuff_->emplace_back(data_ptr);
                input_dims_->emplace_back(dims);
            }else{
                num_output++;
                outBuff_->emplace_back(data_ptr);
                output_dims_->emplace_back(dims);
            }
            // Print Binding Info
            std::cout 
            << engine_->getBindingName(i) 
            << ": (" << engine_->getBindingFormatDesc(i) << ")\n";
            std::cout << "Binding Dims: \n";
            for(int j; j < dims.nbDims; j++){
                cout << j << " - " << dims.d[j] << "\n";
            }
        }

    }
    void Inferer::BuildContext(){
        std::cout << "Unimplemented\n";
    }
    void Inferer::Infer(){
        // Concatenate the bindings
        std::vector<void*> bnd;
        bnd.reserve(inBuff_.size()+outBuff_.size());
        bnd.insert(bnd.end(), inBuff_.begin(), inBuff_.end());
        bnd.insert(bnd.end(), outBuff_.begin(), outBuff_end());
        // Check Validity
        for (int i; i < bnd.size(); i++){
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
        for(int i = 0; i < inBuff.size(); i++){
            cudaFree(inBuff[i]);
        }
        for(int i = 0; i < outBuff.size(); i++){
            cudaFree(outBuff[i]);
        }

    }
    
}


