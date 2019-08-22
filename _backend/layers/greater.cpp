#include "greater.h"
//cpp stuff
namespace layers {    
   
    Greater::Greater(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\greater.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* Greater::_get_device() {
        
        return backend::device;
    }
    
    void Greater::init() {      
  
    }
    
    void Greater::bind(std::string _A_i, std::string _B_i, std::string _C_o){
        A_i = _A_i; B_i = _B_i; C_o = _C_o;

		//binding.A_i = tensor_dict[A_i]->shape();
  		//binding.B_i = tensor_dict[B_i]->shape();
 
		//binding.C_o = tensor_dict[C_o]->shape();
 
        
    }

    void Greater::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[A_i]->data(), *tensor_dict[B_i]->data(), *tensor_dict[C_o]->data());
    }

}

