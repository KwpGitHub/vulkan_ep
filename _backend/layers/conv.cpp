#include "Conv.h"
//cpp stuff
namespace backend {    
   
    Conv::Conv(const std::string& name) : Layer(name) { }
       
    vuh::Device* Conv::_get_device() {
        
        return device;
    }
    
    void Conv::init( int _auto_pad,  Shape_t _dilations,  int _group,  Shape_t _kernel_shape,  Shape_t _pads,  Shape_t _strides) {      
		 auto_pad = _auto_pad; 
 		 dilations = _dilations; 
 		 group = _group; 
 		 kernel_shape = _kernel_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void Conv::bind(std::string _X_input, std::string _W_input, std::string _B_input_opt, std::string _Y_output){
        X_input = _X_input; W_input = _W_input; B_input_opt = _B_input_opt; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.W_input = tensor_dict[W_input]->shape();
  		binding.B_input_opt = tensor_dict[B_input_opt]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.auto_pad = auto_pad;
  		binding.dilations = dilations;
  		binding.group = group;
  		binding.kernel_shape = kernel_shape;
  		binding.pads = pads;
  		binding.strides = strides;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/conv.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[W_input]->data(), *tensor_dict[B_input_opt]->data(), *tensor_dict[Y_output]->data());
    }

}

