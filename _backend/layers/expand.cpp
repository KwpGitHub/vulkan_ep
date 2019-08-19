#include "Expand.h"
//cpp stuff
namespace backend {    
   
    Expand::Expand() : Layer() { }
       
    vuh::Device* Expand::_get_device() {
        
        return device;
    }
    
    void Expand::init() {      
  
    }
    
    void Expand::bind(std::string _input_input, std::string _shape_input, std::string _output_output){
        input_input = _input_input; shape_input = _shape_input; output_output = _output_output;
		binding.input_input = tensor_dict[input_input]->shape();
  		binding.shape_input = tensor_dict[shape_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/expand.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[shape_input]->data(), *tensor_dict[output_output]->data());
    }



}



