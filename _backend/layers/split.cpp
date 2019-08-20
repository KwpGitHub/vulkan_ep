#include "Split.h"
//cpp stuff
namespace backend {    
   
    Split::Split(const std::string& name) : Layer(name) { }
       
    vuh::Device* Split::_get_device() {
        
        return device;
    }
    
    void Split::init( int _axis,  Shape_t _split) {      
		 axis = _axis; 
 		 split = _split; 
  
    }
    
    void Split::bind(std::string _input_i){
        input_i = _input_i;
		binding.input_i = tensor_dict[input_i]->shape();
 

		binding.axis = axis;
  		binding.split = split;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/split.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_i]->data());
    }

}

