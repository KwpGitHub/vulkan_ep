#include "LinearRegressor.h"
//cpp stuff
namespace backend {    
   
    LinearRegressor::LinearRegressor() : Layer() { }
       
    vuh::Device* LinearRegressor::_get_device() {
        
        return device;
    }
    
    void LinearRegressor::init( int _post_transform,  int _targets) {      
		 post_transform = _post_transform; 
 		 targets = _targets; 
  
    }
    
    void LinearRegressor::bind(std::string _coefficients, std::string _intercepts, std::string _X_input, std::string _Y_output){
        coefficients = _coefficients; intercepts = _intercepts; X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.post_transform = post_transform;
  		binding.targets = targets;
 
		binding.coefficients = tensor_dict[coefficients]->shape();
  		binding.intercepts = tensor_dict[intercepts]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/linearregressor.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[coefficients]->data(), *tensor_dict[intercepts]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }



}



