#include "RoiAlign.h"

//cpp stuff
namespace backend {    
   
    RoiAlign::RoiAlign(std::string n, int mode, int output_height, int output_width, int sampling_ratio, float spatial_scale) : Layer(n) { }
       
    vuh::Device* RoiAlign::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void RoiAlign::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.rois_input = tensor_dict[rois_input]->shape();
  		binding.batch_indices_input = tensor_dict[batch_indices_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.mode = mode;
  		binding.output_height = output_height;
  		binding.output_width = output_width;
  		binding.sampling_ratio = sampling_ratio;
  		binding.spatial_scale = spatial_scale;
 
    }
    
    void RoiAlign::call(std::string X_input, std::string rois_input, std::string batch_indices_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/roialign.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[rois_input]->data(), *tensor_dict[batch_indices_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


