#include "LinearRegressor.h"
//cpp stuff
namespace backend {    
   
    LinearRegressor::LinearRegressor(const std::string& name) : Layer(name) { }
       
    vuh::Device* LinearRegressor::_get_device() {
        
        return device;
    }
    
    void LinearRegressor::init( int _post_transform,  int _targets) {      
		 post_transform = _post_transform; 
 		 targets = _targets; 
  
    }
    
    void LinearRegressor::bind(std::string _coefficients, std::string _intercepts, std::string _X_i, std::string _Y_o){
        coefficients = _coefficients; intercepts = _intercepts; X_i = _X_i; Y_o = _Y_o;
		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.post_transform = post_transform;
  		binding.targets = targets;
 
		binding.coefficients = tensor_dict[coefficients]->shape();
  		binding.intercepts = tensor_dict[intercepts]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/linearregressor.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[coefficients]->data(), *tensor_dict[intercepts]->data(), *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

