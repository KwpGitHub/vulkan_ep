#include "Loop.h"
//cpp stuff
namespace backend {    
   
    Loop::Loop(std::string name) : Layer(name) { }
       
    vuh::Device* Loop::_get_device() {
        
        return device;
    }
    
    void Loop::init( int _body) {      
		 body = _body; 
  
    }
    
    void Loop::bind(std::string _M_i, std::string _cond_i){
        M_i = _M_i; cond_i = _cond_i;

		binding.M_i = tensor_dict[M_i]->shape();
  		binding.cond_i = tensor_dict[cond_i]->shape();
 

		binding.body = body;
 

        
    }
}

