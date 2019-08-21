#include "If.h"
//cpp stuff
namespace backend {    
   
    If::If(std::string name) : Layer(name) { }
       
    vuh::Device* If::_get_device() {
        
        return device;
    }
    
    void If::init( int _else_branch,  int _then_branch) {      
		 else_branch = _else_branch; 
 		 then_branch = _then_branch; 
  
    }
    
    void If::bind(std::string _cond_i){
        cond_i = _cond_i;

		binding.cond_i = tensor_dict[cond_i]->shape();
 

		binding.else_branch = else_branch;
  		binding.then_branch = then_branch;
 

        
    }
}

