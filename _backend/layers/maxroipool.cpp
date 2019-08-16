#include "MaxRoiPool.h"

//cpp stuff
namespace backend {    
   
    MaxRoiPool::MaxRoiPool(std::string n) : Layer(n) { }
       
    vuh::Device* MaxRoiPool::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void MaxRoiPool::init( Shape_t _pooled_shape,  float _spatial_scale) {      
		 pooled_shape = _pooled_shape; 
 		 spatial_scale = _spatial_scale; 
  
    }
    
    void MaxRoiPool::bind(std::string _X_input, std::string _rois_input, std::string _Y_output){
        X_input = _X_input; rois_input = _rois_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.rois_input = tensor_dict[rois_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.pooled_shape = pooled_shape;
  		binding.spatial_scale = spatial_scale;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/maxroipool.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[rois_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    //backend::nn;

//python stuff


