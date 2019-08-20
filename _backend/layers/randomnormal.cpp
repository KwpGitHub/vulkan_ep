#include "RandomNormal.h"
//cpp stuff
namespace backend {    
   
    RandomNormal::RandomNormal(const std::string& name) : Layer(name) { }
       
    vuh::Device* RandomNormal::_get_device() {
        
        return device;
    }
    
    void RandomNormal::init( Shape_t _shape,  int _dtype,  float _mean,  float _scale,  float _seed) {      
		 shape = _shape; 
 		 dtype = _dtype; 
 		 mean = _mean; 
 		 scale = _scale; 
 		 seed = _seed; 
  
    }
    
    void RandomNormal::bind(std::string _output_output){
        output_output = _output_output;

		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.shape = shape;
  		binding.dtype = dtype;
  		binding.mean = mean;
  		binding.scale = scale;
  		binding.seed = seed;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomnormal.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[output_output]->data());
    }

}

