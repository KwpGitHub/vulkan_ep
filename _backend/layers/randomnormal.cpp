#include "RandomNormal.h"

//cpp stuff
namespace backend {    
   
    RandomNormal::RandomNormal(std::string n, Shape_t shape, int dtype, float mean, float scale, float seed) : Layer(n) { }
       
    vuh::Device* RandomNormal::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void RandomNormal::init() {      
    

		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.shape = shape;
  		binding.dtype = dtype;
  		binding.mean = mean;
  		binding.scale = scale;
  		binding.seed = seed;
 
    }
    
    void RandomNormal::call(std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/randomnormal.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[output_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


