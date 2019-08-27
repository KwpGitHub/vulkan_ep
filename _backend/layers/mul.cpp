#include "mul.h"
//cpp stuff
namespace layers {    
   
    Mul::Mul(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/mul.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Mul::_get_device() {        
        return backend::device;
    }
    
    void Mul::init() {      
  
    }
    
    void Mul::bind(std::string _A_i, std::string _B_i, std::string _C_o){
        A_i = _A_i; B_i = _B_i; C_o = _C_o;

		binding.A_i = backend::tensor_dict[A_i]->shape();
  		binding.B_i = backend::tensor_dict[B_i]->shape();
 
		binding.C_o = backend::tensor_dict[C_o]->shape();
 
        
    }

    void Mul::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[A_i]->data(), *backend::tensor_dict[B_i]->data(), *backend::tensor_dict[C_o]->data());
    }

    void Mul::forward(){ 
        program->run();
    }

}

