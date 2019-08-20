#include "And.h"
//cpp stuff
namespace backend {    
   
    And::And(const std::string& name) : Layer(name) { }
       
    vuh::Device* And::_get_device() {
        
        return device;
    }
    
    void And::init() {      
  
    }
    
    void And::bind(std::string _A_i, std::string _B_i, std::string _C_o){
        A_i = _A_i; B_i = _B_i; C_o = _C_o;
		binding.A_i = tensor_dict[A_i]->shape();
  		binding.B_i = tensor_dict[B_i]->shape();
 
		binding.C_o = tensor_dict[C_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/and.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[A_i]->data(), *tensor_dict[B_i]->data(), *tensor_dict[C_o]->data());
    }

}

