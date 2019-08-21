#include "Xor.h"
//cpp stuff
namespace backend {    
   
    Xor::Xor(std::string name) : Layer(name) { }
       
    vuh::Device* Xor::_get_device() {
        
        return device;
    }
    
    void Xor::init() {      
  
    }
    
    void Xor::bind(std::string _A_i, std::string _B_i, std::string _C_o){
        A_i = _A_i; B_i = _B_i; C_o = _C_o;

		binding.A_i = tensor_dict[A_i]->shape();
  		binding.B_i = tensor_dict[B_i]->shape();
 
		binding.C_o = tensor_dict[C_o]->shape();
 


        
    }
}

