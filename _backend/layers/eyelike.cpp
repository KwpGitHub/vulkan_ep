#include "EyeLike.h"
//cpp stuff
namespace backend {    
   
    EyeLike::EyeLike(const std::string& name) : Layer(name) { }
       
    vuh::Device* EyeLike::_get_device() {
        
        return device;
    }
    
    void EyeLike::init( int _dtype,  int _k) {      
		 dtype = _dtype; 
 		 k = _k; 
  
    }
    
    void EyeLike::bind(std::string _input_input, std::string _output_output){
        input_input = _input_input; output_output = _output_output;
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.dtype = dtype;
  		binding.k = k;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/eyelike.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }

}

