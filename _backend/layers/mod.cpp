#include "Mod.h"
//cpp stuff
namespace backend {    
   
    Mod::Mod(const std::string& name) : Layer(name) { }
       
    vuh::Device* Mod::_get_device() {
        
        return device;
    }
    
    void Mod::init( int _fmod) {      
		 fmod = _fmod; 
  
    }
    
    void Mod::bind(std::string _A_i, std::string _B_i, std::string _C_o){
        A_i = _A_i; B_i = _B_i; C_o = _C_o;
		binding.A_i = tensor_dict[A_i]->shape();
  		binding.B_i = tensor_dict[B_i]->shape();
 
		binding.C_o = tensor_dict[C_o]->shape();
 
		binding.fmod = fmod;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/mod.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[A_i]->data(), *tensor_dict[B_i]->data(), *tensor_dict[C_o]->data());
    }

}
