#include "LinearRegressor.h"
//cpp stuff
namespace backend {    
   
    LinearRegressor::LinearRegressor(std::string name) : Layer(name) { }
       
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
 
        
    }
}

