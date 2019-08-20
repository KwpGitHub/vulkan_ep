#include "MaxUnpool.h"
//cpp stuff
namespace backend {    
   
    MaxUnpool::MaxUnpool(const std::string& name) : Layer(name) { }
       
    vuh::Device* MaxUnpool::_get_device() {
        
        return device;
    }
    
    void MaxUnpool::init( Shape_t _kernel_shape,  Shape_t _pads,  Shape_t _strides) {      
		 kernel_shape = _kernel_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void MaxUnpool::bind(std::string _X_input, std::string _I_input, std::string _output_shape_input_opt, std::string _output_output){
        X_input = _X_input; I_input = _I_input; output_shape_input_opt = _output_shape_input_opt; output_output = _output_output;
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.I_input = tensor_dict[I_input]->shape();
  		binding.output_shape_input_opt = tensor_dict[output_shape_input_opt]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.kernel_shape = kernel_shape;
  		binding.pads = pads;
  		binding.strides = strides;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/maxunpool.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[I_input]->data(), *tensor_dict[output_shape_input_opt]->data(), *tensor_dict[output_output]->data());
    }

}

