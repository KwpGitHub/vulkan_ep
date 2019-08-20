#include "RoiAlign.h"
//cpp stuff
namespace backend {    
   
    RoiAlign::RoiAlign(const std::string& name) : Layer(name) { }
       
    vuh::Device* RoiAlign::_get_device() {
        
        return device;
    }
    
    void RoiAlign::init( int _mode,  int _output_height,  int _output_width,  int _sampling_ratio,  float _spatial_scale) {      
		 mode = _mode; 
 		 output_height = _output_height; 
 		 output_width = _output_width; 
 		 sampling_ratio = _sampling_ratio; 
 		 spatial_scale = _spatial_scale; 
  
    }
    
    void RoiAlign::bind(std::string _X_i, std::string _rois_i, std::string _batch_indices_i, std::string _Y_o){
        X_i = _X_i; rois_i = _rois_i; batch_indices_i = _batch_indices_i; Y_o = _Y_o;
		binding.X_i = tensor_dict[X_i]->shape();
  		binding.rois_i = tensor_dict[rois_i]->shape();
  		binding.batch_indices_i = tensor_dict[batch_indices_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.mode = mode;
  		binding.output_height = output_height;
  		binding.output_width = output_width;
  		binding.sampling_ratio = sampling_ratio;
  		binding.spatial_scale = spatial_scale;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/roialign.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[rois_i]->data(), *tensor_dict[batch_indices_i]->data(), *tensor_dict[Y_o]->data());
    }

}

