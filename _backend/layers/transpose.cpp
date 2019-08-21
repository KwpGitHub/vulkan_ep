#include "Transpose.h"
//cpp stuff
namespace backend {    
   
    Transpose::Transpose(std::string name) : Layer(name) { }
       
    vuh::Device* Transpose::_get_device() {
        
        return device;
    }
    
    void Transpose::init( Shape_t _perm) {      
		 perm = _perm; 
  
    }
    
    void Transpose::bind(std::string _data_i, std::string _transposed_o){
        data_i = _data_i; transposed_o = _transposed_o;

		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.transposed_o = tensor_dict[transposed_o]->shape();
 
		binding.perm = perm;
 

        
    }
}

