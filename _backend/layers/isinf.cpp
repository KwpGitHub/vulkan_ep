#include "IsInf.h"
//cpp stuff
namespace backend {    
   
    IsInf::IsInf(std::string name) : Layer(name) { }
       
    vuh::Device* IsInf::_get_device() {
        
        return device;
    }
    
    void IsInf::init( int _detect_negative,  int _detect_positive) {      
		 detect_negative = _detect_negative; 
 		 detect_positive = _detect_positive; 
  
    }
    
    void IsInf::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.detect_negative = detect_negative;
  		binding.detect_positive = detect_positive;
 

        
    }
}

