#include "LinearRegressor.h"

//cpp stuff
namespace backend {    
   
    LinearRegressor::LinearRegressor(std::string n, int post_transform, int targets) : Layer(n) { }
       
    vuh::Device* LinearRegressor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void LinearRegressor::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.post_transform = post_transform;
  		binding.targets = targets;
  		binding.coefficients = tensor_dict[coefficients]->shape();
  		binding.intercepts = tensor_dict[intercepts]->shape();
 
    }
    
    void LinearRegressor::call(std::string coefficients, std::string intercepts, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/linearregressor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[coefficients]->data(), *tensor_dict[intercepts]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }
    
}

    py::module m("_backend.nn", "nn MOD");

//python stuff


