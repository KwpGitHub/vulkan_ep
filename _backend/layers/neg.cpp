#include "Neg.h"
//cpp stuff
namespace backend {    
   
    Neg::Neg(std::string name) : Layer(name) { }
       
    vuh::Device* Neg::_get_device() {
        
        return device;
    }
    
    void Neg::init() {      
  
    }
    
    void Neg::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 


        
    }
}

