#include "Split.h"
//cpp stuff
namespace backend {    
   
    Split::Split() : Layer() { }
       
    vuh::Device* Split::_get_device() {
        
        return device;
    }
    
    void Split::init( int _axis,  Shape_t _split) {      
		 axis = _axis; 
 		 split = _split; 
  
    }
    
    void Split::bind(std::string _input_input){
        input_input = _input_input;
		binding.input_input = tensor_dict[input_input]->shape();
 

		binding.axis = axis;
  		binding.split = split;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/split.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_input]->data());
    }



}



