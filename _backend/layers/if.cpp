#include "if.h"
//cpp stuff
namespace layers {    
   
    If::If(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/if.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* If::_get_device() {
        
        return backend::device;
    }
    
    void If::init( int _else_branch,  int _then_branch) {      
		 else_branch = _else_branch; 
 		 then_branch = _then_branch; 
  
    }
    
    void If::bind(std::string _cond_i){
        cond_i = _cond_i;

		//binding.cond_i = tensor_dict[cond_i]->shape();
 

		//binding.else_branch = else_branch;
  		//binding.then_branch = then_branch;
         
    }

    void If::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[cond_i]->data());
    }

}

