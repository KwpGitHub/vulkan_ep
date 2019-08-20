#include "If.h"
//cpp stuff
namespace backend {    
   
    If::If(const std::string& name) : Layer(name) { }
       
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
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/if.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[cond_i]->data());
    }

}

