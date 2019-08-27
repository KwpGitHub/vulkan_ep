#include "linearregressor.h"
//cpp stuff
namespace layers {    
   
    LinearRegressor::LinearRegressor(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/linearregressor.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* LinearRegressor::_get_device() {        
        return backend::device;
    }
    
    void LinearRegressor::init( std::vector<float> _coefficients,  std::vector<float> _intercepts,  std::string _post_transform,  int _targets) {      
		 coefficients = _coefficients; 
 		 intercepts = _intercepts; 
 		 post_transform = _post_transform; 
 		 targets = _targets; 
  
    }
    
    void LinearRegressor::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.coefficients = coefficients;
  		//binding.intercepts = intercepts;
  		//binding.post_transform = post_transform;
  		//binding.targets = targets;
         
    }

    void LinearRegressor::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void LinearRegressor::forward(){ 
        //program->run();
    }

}

