#include "Slice.h"
//cpp stuff
namespace backend {    
   
    Slice::Slice(const std::string& name) : Layer(name) { }
       
    vuh::Device* Slice::_get_device() {
        
        return device;
    }
    
    void Slice::init() {      
  
    }
    
    void Slice::bind(std::string _data_i, std::string _starts_i, std::string _ends_i, std::string _axes_i, std::string _steps_i, std::string _output_o){
        data_i = _data_i; starts_i = _starts_i; ends_i = _ends_i; axes_i = _axes_i; steps_i = _steps_i; output_o = _output_o;
		binding.data_i = tensor_dict[data_i]->shape();
  		binding.starts_i = tensor_dict[starts_i]->shape();
  		binding.ends_i = tensor_dict[ends_i]->shape();
  		binding.axes_i = tensor_dict[axes_i]->shape();
  		binding.steps_i = tensor_dict[steps_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/slice.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[starts_i]->data(), *tensor_dict[ends_i]->data(), *tensor_dict[axes_i]->data(), *tensor_dict[steps_i]->data(), *tensor_dict[output_o]->data());
    }

}

