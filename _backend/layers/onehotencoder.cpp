#include "OneHotEncoder.h"
//cpp stuff
namespace backend {    
   
    OneHotEncoder::OneHotEncoder(std::string name) : Layer(name) { }
       
    vuh::Device* OneHotEncoder::_get_device() {
        
        return device;
    }
    
    void OneHotEncoder::init( Shape_t _cats_int64s,  int _zeros) {      
		 cats_int64s = _cats_int64s; 
 		 zeros = _zeros; 
  
    }
    
    void OneHotEncoder::bind(std::string _cats_strings, std::string _X_i, std::string _Y_o){
        cats_strings = _cats_strings; X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.cats_int64s = cats_int64s;
  		binding.zeros = zeros;
 
		binding.cats_strings = tensor_dict[cats_strings]->shape();
 
        
    }
}

