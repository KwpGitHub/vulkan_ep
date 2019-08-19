#include "LpNormalization.h"
//cpp stuff
namespace backend {    
   
    LpNormalization::LpNormalization() : Layer() { }
       
    vuh::Device* LpNormalization::_get_device() {
        
        return device;
    }
    
    void LpNormalization::init( int _axis,  int _p) {      
		 axis = _axis; 
 		 p = _p; 
  
    }
    
    void LpNormalization::bind(std::string _input_input, std::string _output_output){
        input_input = _input_input; output_output = _output_output;
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.axis = axis;
  		binding.p = p;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/lpnormalization.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }



}



