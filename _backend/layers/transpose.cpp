#include "Transpose.h"
//cpp stuff
namespace backend {    
   
    Transpose::Transpose(const std::string& name) : Layer(name) { }
       
    vuh::Device* Transpose::_get_device() {
        
        return device;
    }
    
    void Transpose::init( Shape_t _perm) {      
		 perm = _perm; 
  
    }
    
    void Transpose::bind(std::string _data_i, std::string _transposed_o){
        data_i = _data_i; transposed_o = _transposed_o;
		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.transposed_o = tensor_dict[transposed_o]->shape();
 
		binding.perm = perm;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/transpose.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[transposed_o]->data());
    }

}

