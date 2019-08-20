#include "Loop.h"
//cpp stuff
namespace backend {    
   
    Loop::Loop(const std::string& name) : Layer(name) { }
       
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
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/loop.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[M_i]->data(), *tensor_dict[cond_i]->data());
    }

}

