#include "RandomUniform.h"
//cpp stuff
namespace backend {    
   
    RandomUniform::RandomUniform(const std::string& name) : Layer(name) { }
       
    vuh::Device* RandomUniform::_get_device() {
        
        return device;
    }
    
    void RandomUniform::init( Shape_t _shape,  int _dtype,  float _high,  float _low,  float _seed) {      
		 shape = _shape; 
 		 dtype = _dtype; 
 		 high = _high; 
 		 low = _low; 
 		 seed = _seed; 
  
    }
    
    void RandomUniform::bind(std::string _output_o){
        output_o = _output_o;

		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.shape = shape;
  		binding.dtype = dtype;
  		binding.high = high;
  		binding.low = low;
  		binding.seed = seed;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomuniform.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[output_o]->data());
    }

}

