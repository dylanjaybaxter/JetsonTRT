/*
File: yolov8.cpp
Author: Dylan Baxter
Date: 3/8/23
Description: 
    This file contains definition for functions and
    variables necessary for the functioning of yolov8
    in tensorrt.
*/

#include "yolov8.hpp"


namespace jetsontrt::yolov8 {
    struct bbox{
        int cls_id;
        std::string cls_name;
        float conf;
        int x;
        int y;
        int w;
        int h;
    }
    Yolov8::Yolov8(const Configuration config) : Inferer(config){
        // This function serves to explicitly call the generic constructor
    }

    std::vector<bboxes> Inferer::RunYOLOv8(cv::Mat im){
        // Preprocess Image
        preprocess(im, inBuff_);

        // Run Inference
        Infer();

        // Post Process Output
        return postprocess(outBuff_);
    }

    void Yolov8::preprocess(const cv::Mat &im, void* input_buffer){
        // Get Network Input Size
        net_h = input_dims_.at(0).d[2];
        net_w = input_dims_.at(0).d[3];
        
        // Read image size
        input_size_[0] = im.rows;
        input_size_[1] = im.cols;

        // Preallocate Mats
        cv::Mat resize_mat(net_h, net_w, cv::CV_8UC3);
        cv::Mat color_mat(net_h, net_w, cv::CV_8UC3);
        cv::Mat norm_mat(net_h, net_w, cv::CV_32FC3);

        // Resize Image
        cv::resize(im, resize_mat, resize_mat.size(), 0,0, cv::INTER_LINEAR);

        // Convert image from BGR to RGB
        cv::cvtColor(resize_mat, rgb_mat, cv::COLOR_BGR2RGB);

        // Split the image into channels
        rgb_mat.convertTo(norm_mat, cv::CV_32FC3, 1./255.f);

        // Save each channel in series to the input buffer
        std::vector<cv::Mat> ch(3);
        cv::split(norm_mat, ch);
        assert(inBuff.at(0));
        int matSize = net_h*net_h*sizeof(_Float32)''
        for(int i;i < ch.size(); i++){
            CUDA(cudaMemcpy(reinterpret_cast<_Float32*>(inBuff_.at(0))+(matSize*i),
            ch[i].data,
            matSize,
            cudaMemcpyHostToDevice));
        }
    }

    std::vector<bbox> YOLOv8::postprocess(){
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
            sizeof(int32_t),
            cudaMemcpyDeviceToHost));
        
        CUDA(cudaMemcpy(reinterpret_cast<_Float32*>(&bbox_data[0]),
            reinterpret_cast<_Float32*>(outBuff_.at(1)),
            bbox_size,
            cudaMemcpyDeviceToHost));

        CUDA(cudaMemcpy(reinterpret_cast<_Float32*>(&scores[0]),
            reinterpret_cast<_Float32*>(outBuff_.at(2)),
            other_size,
            cudaMemcpyDeviceToHost));
        
        CUDA(cudaMemcpy(reinterpret_cast<_Float32*>(&labels[0]),
            reinterpret_cast<_Float32*>(outBuff_.at(3)),
            other_size,
            cudaMemcpyDeviceToHost));

        // Process output blobs into a list of bboxes
        std::vector<bbox> bbox_list;
        // Bboxes are scaled by input size determine conversion factor
        float fx = (float)input_size[0]/(float)input_dims_[0].d[2];
        float fy = (float)input_size[1]/(float)input_dims_[0].d[3];
        //Now extract bboxes in format (x1, y1, x2, y2)
        for (int i=0; i < nbDet; i++){
            // Initialize
            bbox box;
            // Capture variables
            box.x = bbox_data[i*4]*fx;
            box.y = bbox_data[(i*4)+1]*fy;
            box.w = (bbox_data[(i*4)+2]-bbox_data[(i*4)])*fx;
            box.h = (bbox_data[(i*4)+3]-bbox_data[(i*4)+1])*fy;
            box.conf = scores[i];
            box.cls_id = labels[i];
            if(kNameMap.find(box.cls_id) != kNameMap.end()){
                box.cls_name = kNameMap[box.cls_id];
            }else{
                box.cls_name = "Unimplemented";
            }
        }

    }
}