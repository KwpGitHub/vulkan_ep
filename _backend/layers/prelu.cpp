#include "PRelu.h"

//cpp stuff
namespace backend {    
   
    PRelu::PRelu(std::string n) : Layer(n) { }
       
    vuh::Device* PRelu::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void PRelu::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.slope_input = tensor_dict[slope_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 

    }
    
    void PRelu::call(std::string X_input, std::string slope_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/prelu.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[slope_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


