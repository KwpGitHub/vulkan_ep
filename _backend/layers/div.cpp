#include "div.h"
//cpp stuff
namespace layers {    
   
    Div::Div(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/div.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Div::_get_device() {        
        return backend::device;
    }
    
    void Div::init() {      
  
    }
    
    void Div::bind(std::string _A_i, std::string _B_i, std::string _C_o){
        A_i = _A_i; B_i = _B_i; C_o = _C_o;

		binding.A_i = backend::tensor_dict[A_i]->shape();
  		binding.B_i = backend::tensor_dict[B_i]->shape();
 
		binding.C_o = backend::tensor_dict[C_o]->shape();
 
        
    }

    void Div::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[A_i]->data(), *backend::tensor_dict[B_i]->data(), *backend::tensor_dict[C_o]->data());
    }

    void Div::forward(){ 
        program->run();
    }

}

