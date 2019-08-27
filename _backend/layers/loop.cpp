#include "loop.h"
//cpp stuff
namespace layers {    
   
    Loop::Loop(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/loop.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Loop::_get_device() {        
        return backend::device;
    }
    
    void Loop::init( int _body) {      
		 body = _body; 
  
    }
    
    void Loop::bind(std::string _M_i, std::string _cond_i){
        M_i = _M_i; cond_i = _cond_i;

		binding.M_i = backend::tensor_dict[M_i]->shape();
  		binding.cond_i = backend::tensor_dict[cond_i]->shape();
 

		//binding.body = body;
         
    }

    void Loop::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[M_i]->data(), *backend::tensor_dict[cond_i]->data());
    }

    void Loop::forward(){ 
        program->run();
    }

}

