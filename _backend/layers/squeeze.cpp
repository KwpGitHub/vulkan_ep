#include "Squeeze.h"
//cpp stuff
namespace backend {    
   
    Squeeze::Squeeze(const std::string& name) : Layer(name) { }
       
    vuh::Device* Squeeze::_get_device() {
        
        return device;
    }
    
    void Squeeze::init( Shape_t _axes) {      
		 axes = _axes; 
  
    }
    
    void Squeeze::bind(std::string _data_i, std::string _squeezed_o){
        data_i = _data_i; squeezed_o = _squeezed_o;
		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.squeezed_o = tensor_dict[squeezed_o]->shape();
 
		binding.axes = axes;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/squeeze.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[squeezed_o]->data());
    }

}

