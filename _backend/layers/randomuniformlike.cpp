#include "RandomUniformLike.h"

//cpp stuff
namespace backend {    
   
    RandomUniformLike::RandomUniformLike(std::string n, int dtype, float high, float low, float seed) : Layer(n) { }
       
    vuh::Device* RandomUniformLike::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void RandomUniformLike::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.dtype = dtype;
  		binding.high = high;
  		binding.low = low;
  		binding.seed = seed;
 
    }
    
    void RandomUniformLike::call(std::string input_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomuniformlike.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


