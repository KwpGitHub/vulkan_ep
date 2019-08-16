#include "RandomUniform.h"

//cpp stuff
namespace backend {    
   
    RandomUniform::RandomUniform(std::string n) : Layer(n) { }
       
    vuh::Device* RandomUniform::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void RandomUniform::init( Shape_t _shape,  int _dtype,  float _high,  float _low,  float _seed) {      
		 shape = _shape; 
 		 dtype = _dtype; 
 		 high = _high; 
 		 low = _low; 
 		 seed = _seed; 
  
    }
    
    void RandomUniform::bind(std::string _output_output){
        output_output = _output_output;

		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.shape = shape;
  		binding.dtype = dtype;
  		binding.high = high;
  		binding.low = low;
  		binding.seed = seed;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomuniform.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[output_output]->data());
    }
    
}

    //backend::nn;

//python stuff


