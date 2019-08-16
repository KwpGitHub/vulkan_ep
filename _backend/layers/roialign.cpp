#include "RoiAlign.h"

//cpp stuff
namespace backend {    
   
    RoiAlign::RoiAlign(std::string n) : Layer(n) { }
       
    vuh::Device* RoiAlign::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void RoiAlign::init( int _mode,  int _output_height,  int _output_width,  int _sampling_ratio,  float _spatial_scale) {      
		 mode = _mode; 
 		 output_height = _output_height; 
 		 output_width = _output_width; 
 		 sampling_ratio = _sampling_ratio; 
 		 spatial_scale = _spatial_scale; 
  
    }
    
    void RoiAlign::bind(std::string _X_input, std::string _rois_input, std::string _batch_indices_input, std::string _Y_output){
        X_input = _X_input; rois_input = _rois_input; batch_indices_input = _batch_indices_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.rois_input = tensor_dict[rois_input]->shape();
  		binding.batch_indices_input = tensor_dict[batch_indices_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.mode = mode;
  		binding.output_height = output_height;
  		binding.output_width = output_width;
  		binding.sampling_ratio = sampling_ratio;
  		binding.spatial_scale = spatial_scale;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/roialign.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[rois_input]->data(), *tensor_dict[batch_indices_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    //backend::nn;

//python stuff


