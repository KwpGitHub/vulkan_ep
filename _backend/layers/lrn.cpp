#include "LRN.h"
//cpp stuff
namespace backend {    
   
    LRN::LRN(std::string name) : Layer(name) { }
       
    vuh::Device* LRN::_get_device() {
        
        return device;
    }
    
    void LRN::init( int _size,  float _alpha,  float _beta,  float _bias) {      
		 size = _size; 
 		 alpha = _alpha; 
 		 beta = _beta; 
 		 bias = _bias; 
  
    }
    
    void LRN::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.size = size;
  		binding.alpha = alpha;
  		binding.beta = beta;
  		binding.bias = bias;
 

        
    }
}

