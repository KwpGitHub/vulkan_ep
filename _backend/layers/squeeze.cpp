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
    
    void Squeeze::bind(std::string _data_input, std::string _squeezed_output){
        data_input = _data_input; squeezed_output = _squeezed_output;
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.squeezed_output = tensor_dict[squeezed_output]->shape();
 
		binding.axes = axes;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/squeeze.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[squeezed_output]->data());
    }

}

