#include "LpPool.h"
//cpp stuff
namespace backend {    
   
    LpPool::LpPool() : Layer() { }
       
    vuh::Device* LpPool::_get_device() {
        
        return device;
    }
    
    void LpPool::init( Shape_t _kernel_shape,  int _auto_pad,  int _p,  Shape_t _pads,  Shape_t _strides) {      
		 kernel_shape = _kernel_shape; 
 		 auto_pad = _auto_pad; 
 		 p = _p; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void LpPool::bind(std::string _X_input, std::string _Y_output){
        X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.kernel_shape = kernel_shape;
  		binding.auto_pad = auto_pad;
  		binding.p = p;
  		binding.pads = pads;
  		binding.strides = strides;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/lppool.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }



}



