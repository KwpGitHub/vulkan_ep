#include "or.h"
//cpp stuff
namespace layers {    
   
    Or::Or(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/or.spv");
       
        //program = new vuh::Program<Specs, Params>(*_get_device(), std::string(std::string(backend::file_path) + std::string("saxpy.spv")).c_str());

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Or::_get_device() {
        
        return backend::device;
    }
    
    void Or::init() {      
  
    }
    
    void Or::bind(std::string _A_i, std::string _B_i, std::string _C_o){
        A_i = _A_i; B_i = _B_i; C_o = _C_o;

		//binding.A_i = tensor_dict[A_i]->shape();
  		//binding.B_i = tensor_dict[B_i]->shape();
 
		//binding.C_o = tensor_dict[C_o]->shape();
 
        
    }

    void Or::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[A_i]->data(), *tensor_dict[B_i]->data(), *tensor_dict[C_o]->data());
    }

}

