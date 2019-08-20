#include "RandomNormalLike.h"
//cpp stuff
namespace backend {    
   
    RandomNormalLike::RandomNormalLike(const std::string& name) : Layer(name) { }
       
    vuh::Device* RandomNormalLike::_get_device() {
        
        return device;
    }
    
    void RandomNormalLike::init( int _dtype,  float _mean,  float _scale,  float _seed) {      
		 dtype = _dtype; 
 		 mean = _mean; 
 		 scale = _scale; 
 		 seed = _seed; 
  
    }
    
    void RandomNormalLike::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;
		binding.input_i = tensor_dict[input_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.dtype = dtype;
  		binding.mean = mean;
  		binding.scale = scale;
  		binding.seed = seed;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomnormallike.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[output_o]->data());
    }

}

