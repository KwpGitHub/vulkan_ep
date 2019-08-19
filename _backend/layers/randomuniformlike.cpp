#include "RandomUniformLike.h"
//cpp stuff
namespace backend {    
   
    RandomUniformLike::RandomUniformLike() : Layer() { }
       
    vuh::Device* RandomUniformLike::_get_device() {
        
        return device;
    }
    
    void RandomUniformLike::init( int _dtype,  float _high,  float _low,  float _seed) {      
		 dtype = _dtype; 
 		 high = _high; 
 		 low = _low; 
 		 seed = _seed; 
  
    }
    
    void RandomUniformLike::bind(std::string _input_input, std::string _output_output){
        input_input = _input_input; output_output = _output_output;
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.dtype = dtype;
  		binding.high = high;
  		binding.low = low;
  		binding.seed = seed;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomuniformlike.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }



}



