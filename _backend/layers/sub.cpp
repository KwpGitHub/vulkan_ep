#include "Sub.h"
//cpp stuff
namespace backend {    
   
    Sub::Sub() : Layer() { }
       
    vuh::Device* Sub::_get_device() {
        
        return device;
    }
    
    void Sub::init() {      
  
    }
    
    void Sub::bind(std::string _A_input, std::string _B_input, std::string _C_output){
        A_input = _A_input; B_input = _B_input; C_output = _C_output;
		binding.A_input = tensor_dict[A_input]->shape();
  		binding.B_input = tensor_dict[B_input]->shape();
 
		binding.C_output = tensor_dict[C_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/sub.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[A_input]->data(), *tensor_dict[B_input]->data(), *tensor_dict[C_output]->data());
    }



}



