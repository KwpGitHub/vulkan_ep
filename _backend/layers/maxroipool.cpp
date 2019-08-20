#include "MaxRoiPool.h"
//cpp stuff
namespace backend {    
   
    MaxRoiPool::MaxRoiPool(const std::string& name) : Layer(name) { }
       
    vuh::Device* MaxRoiPool::_get_device() {
        
        return device;
    }
    
    void MaxRoiPool::init( Shape_t _pooled_shape,  float _spatial_scale) {      
		 pooled_shape = _pooled_shape; 
 		 spatial_scale = _spatial_scale; 
  
    }
    
    void MaxRoiPool::bind(std::string _X_i, std::string _rois_i, std::string _Y_o){
        X_i = _X_i; rois_i = _rois_i; Y_o = _Y_o;
		binding.X_i = tensor_dict[X_i]->shape();
  		binding.rois_i = tensor_dict[rois_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.pooled_shape = pooled_shape;
  		binding.spatial_scale = spatial_scale;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/maxroipool.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[rois_i]->data(), *tensor_dict[Y_o]->data());
    }

}

