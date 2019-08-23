#include "roialign.h"
//cpp stuff
namespace layers {    
   
    RoiAlign::RoiAlign(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/roialign.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* RoiAlign::_get_device() {
        
        return backend::device;
    }
    
    void RoiAlign::init( std::string _mode,  int _output_height,  int _output_width,  int _sampling_ratio,  float _spatial_scale) {      
		 mode = _mode; 
 		 output_height = _output_height; 
 		 output_width = _output_width; 
 		 sampling_ratio = _sampling_ratio; 
 		 spatial_scale = _spatial_scale; 
  
    }
    
    void RoiAlign::bind(std::string _X_i, std::string _rois_i, std::string _batch_indices_i, std::string _Y_o){
        X_i = _X_i; rois_i = _rois_i; batch_indices_i = _batch_indices_i; Y_o = _Y_o;

		//binding.X_i = tensor_dict[X_i]->shape();
  		//binding.rois_i = tensor_dict[rois_i]->shape();
  		//binding.batch_indices_i = tensor_dict[batch_indices_i]->shape();
 
		//binding.Y_o = tensor_dict[Y_o]->shape();
 
		//binding.mode = mode;
  		//binding.output_height = output_height;
  		//binding.output_width = output_width;
  		//binding.sampling_ratio = sampling_ratio;
  		//binding.spatial_scale = spatial_scale;
         
    }

    void RoiAlign::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[rois_i]->data(), *tensor_dict[batch_indices_i]->data(), *tensor_dict[Y_o]->data());
    }

}

