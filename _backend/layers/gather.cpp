#include "Gather.h"
//cpp stuff
namespace backend {    
   
    Gather::Gather(const std::string& name) : Layer(name) { }
       
    vuh::Device* Gather::_get_device() {
        
        return device;
    }
    
    void Gather::init( int _axis) {      
		 axis = _axis; 
  
    }
    
    void Gather::bind(std::string _data_input, std::string _indices_input, std::string _output_output){
        data_input = _data_input; indices_input = _indices_input; output_output = _output_output;
		binding.data_input = tensor_dict[data_input]->shape();
  		binding.indices_input = tensor_dict[indices_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.axis = axis;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/gather.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[indices_input]->data(), *tensor_dict[output_output]->data());
    }

}

