#include "Unsqueeze.h"
//cpp stuff
namespace backend {    
   
    Unsqueeze::Unsqueeze(const std::string& name) : Layer(name) { }
       
    vuh::Device* Unsqueeze::_get_device() {
        
        return device;
    }
    
    void Unsqueeze::init( Shape_t _axes) {      
		 axes = _axes; 
  
    }
    
    void Unsqueeze::bind(std::string _data_input, std::string _expanded_output){
        data_input = _data_input; expanded_output = _expanded_output;
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.expanded_output = tensor_dict[expanded_output]->shape();
 
		binding.axes = axes;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/unsqueeze.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[expanded_output]->data());
    }

}

