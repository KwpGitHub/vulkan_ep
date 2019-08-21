#include "IsNaN.h"
//cpp stuff
namespace backend {    
   
    IsNaN::IsNaN(std::string name) : Layer(name) { }
       
    vuh::Device* IsNaN::_get_device() {
        
        return device;
    }
    
    void IsNaN::init() {      
  
    }
    
    void IsNaN::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 


        
    }
}

