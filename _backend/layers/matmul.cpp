#include "MatMul.h"
//cpp stuff
namespace backend {    
   
    MatMul::MatMul(const std::string& name) : Layer(name) { }
       
    vuh::Device* MatMul::_get_device() {
        
        return device;
    }
    
    void MatMul::init() {      
  
    }
    
    void MatMul::bind(std::string _A_i, std::string _B_i, std::string _Y_o){
        A_i = _A_i; B_i = _B_i; Y_o = _Y_o;
		binding.A_i = tensor_dict[A_i]->shape();
  		binding.B_i = tensor_dict[B_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/matmul.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[A_i]->data(), *tensor_dict[B_i]->data(), *tensor_dict[Y_o]->data());
    }

}

