#include "RandomNormalLike.h"
//cpp stuff
namespace backend {    
   
    RandomNormalLike::RandomNormalLike() : Layer() { }
       
    vuh::Device* RandomNormalLike::_get_device() {
        
        return device;
    }
    
    void RandomNormalLike::init( int _dtype,  float _mean,  float _scale,  float _seed) {      
		 dtype = _dtype; 
 		 mean = _mean; 
 		 scale = _scale; 
 		 seed = _seed; 
  
    }
    
    void RandomNormalLike::bind(std::string _input_input, std::string _output_output){
        input_input = _input_input; output_output = _output_output;
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.dtype = dtype;
  		binding.mean = mean;
  		binding.scale = scale;
  		binding.seed = seed;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomnormallike.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }



}



