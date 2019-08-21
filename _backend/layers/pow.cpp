#include "Pow.h"
//cpp stuff
namespace backend {    
   
    Pow::Pow(std::string name) : Layer(name) { }
       
    vuh::Device* Pow::_get_device() {
        
        return device;
    }
    
    void Pow::init() {      
  
    }
    
    void Pow::bind(std::string _X_i, std::string _Y_i, std::string _Z_o){
        X_i = _X_i; Y_i = _Y_i; Z_o = _Z_o;

		binding.X_i = tensor_dict[X_i]->shape();
  		binding.Y_i = tensor_dict[Y_i]->shape();
 
		binding.Z_o = tensor_dict[Z_o]->shape();
 


        
    }
}

