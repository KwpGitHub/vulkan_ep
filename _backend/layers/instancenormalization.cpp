#include "InstanceNormalization.h"
//cpp stuff
namespace backend {    
   
    InstanceNormalization::InstanceNormalization() : Layer() { }
       
    vuh::Device* InstanceNormalization::_get_device() {
        
        return device;
    }
    
    void InstanceNormalization::init( float _epsilon) {      
		 epsilon = _epsilon; 
  
    }
    
    void InstanceNormalization::bind(std::string _input_input, std::string _scale_input, std::string _B_input, std::string _output_output){
        input_input = _input_input; scale_input = _scale_input; B_input = _B_input; output_output = _output_output;
		binding.input_input = tensor_dict[input_input]->shape();
  		binding.scale_input = tensor_dict[scale_input]->shape();
  		binding.B_input = tensor_dict[B_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.epsilon = epsilon;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/instancenormalization.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[scale_input]->data(), *tensor_dict[B_input]->data(), *tensor_dict[output_output]->data());
    }



}



