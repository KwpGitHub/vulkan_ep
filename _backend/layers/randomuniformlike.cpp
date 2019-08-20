#include "RandomUniformLike.h"
//cpp stuff
namespace backend {    
   
    RandomUniformLike::RandomUniformLike(const std::string& name) : Layer(name) { }
       
    vuh::Device* RandomUniformLike::_get_device() {
        
        return device;
    }
    
    void RandomUniformLike::init( int _dtype,  float _high,  float _low,  float _seed) {      
		 dtype = _dtype; 
 		 high = _high; 
 		 low = _low; 
 		 seed = _seed; 
  
    }
    
    void RandomUniformLike::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;
		binding.input_i = tensor_dict[input_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.dtype = dtype;
  		binding.high = high;
  		binding.low = low;
  		binding.seed = seed;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomuniformlike.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[output_o]->data());
    }

}
