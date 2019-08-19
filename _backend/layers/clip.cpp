#include "Clip.h"
//cpp stuff
namespace backend {    
   
    Clip::Clip() : Layer() { }
       
    vuh::Device* Clip::_get_device() {
        
        return device;
    }
    
    void Clip::init( float _max,  float _min) {      
		 max = _max; 
 		 min = _min; 
  
    }
    
    void Clip::bind(std::string _input_input, std::string _output_output){
        input_input = _input_input; output_output = _output_output;
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.max = max;
  		binding.min = min;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/clip.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }



}



