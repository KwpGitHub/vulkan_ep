#include "Selu.h"
//cpp stuff
namespace backend {    
   
    Selu::Selu(std::string name) : Layer(name) { }
       
    vuh::Device* Selu::_get_device() {
        
        return device;
    }
    
    void Selu::init( float _alpha,  float _gamma) {      
		 alpha = _alpha; 
 		 gamma = _gamma; 
  
    }
    
    void Selu::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.alpha = alpha;
  		binding.gamma = gamma;
 

        
    }
}

