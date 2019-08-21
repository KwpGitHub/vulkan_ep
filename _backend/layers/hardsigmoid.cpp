#include "HardSigmoid.h"
//cpp stuff
namespace backend {    
   
    HardSigmoid::HardSigmoid(std::string name) : Layer(name) { }
       
    vuh::Device* HardSigmoid::_get_device() {
        
        return device;
    }
    
    void HardSigmoid::init( float _alpha,  float _beta) {      
		 alpha = _alpha; 
 		 beta = _beta; 
  
    }
    
    void HardSigmoid::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.alpha = alpha;
  		binding.beta = beta;
 

        
    }
}

