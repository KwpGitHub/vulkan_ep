#include "PRelu.h"
//cpp stuff
namespace backend {    
   
    PRelu::PRelu(std::string name) : Layer(name) { }
       
    vuh::Device* PRelu::_get_device() {
        
        return device;
    }
    
    void PRelu::init() {      
  
    }
    
    void PRelu::bind(std::string _X_i, std::string _slope_i, std::string _Y_o){
        X_i = _X_i; slope_i = _slope_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
  		binding.slope_i = tensor_dict[slope_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 


        
    }
}

