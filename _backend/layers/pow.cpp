#include "Pow.h"
//cpp stuff
namespace backend {    
   
    Pow::Pow(const std::string& name) : Layer(name) { }
       
    vuh::Device* Pow::_get_device() {
        
        return device;
    }
    
    void Pow::init() {      
  
    }
    
    void Pow::bind(std::string _X_i, std::string _Y_i, std::string _Z_o){
        X_i = _X_i; Y_i = _Y_i; Z_o = _Z_o;
		binding.X_i = tensor_dict[X_i]->shape();
  		binding.Y_i = tensor_dict[Y_i]->shape();
 
		binding.Z_o = tensor_dict[Z_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/pow.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_i]->data(), *tensor_dict[Z_o]->data());
    }

}

