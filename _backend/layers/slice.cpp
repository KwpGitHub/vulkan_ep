#include "Slice.h"
//cpp stuff
namespace backend {    
   
    Slice::Slice() : Layer() { }
       
    vuh::Device* Slice::_get_device() {
        
        return device;
    }
    
    void Slice::init() {      
  
    }
    
    void Slice::bind(std::string _data_input, std::string _starts_input, std::string _ends_input, std::string _axes_input_opt, std::string _steps_input_opt, std::string _output_output){
        data_input = _data_input; starts_input = _starts_input; ends_input = _ends_input; axes_input_opt = _axes_input_opt; steps_input_opt = _steps_input_opt; output_output = _output_output;
		binding.data_input = tensor_dict[data_input]->shape();
  		binding.starts_input = tensor_dict[starts_input]->shape();
  		binding.ends_input = tensor_dict[ends_input]->shape();
  		binding.axes_input_opt = tensor_dict[axes_input_opt]->shape();
  		binding.steps_input_opt = tensor_dict[steps_input_opt]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/slice.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[starts_input]->data(), *tensor_dict[ends_input]->data(), *tensor_dict[axes_input_opt]->data(), *tensor_dict[steps_input_opt]->data(), *tensor_dict[output_output]->data());
    }



}



