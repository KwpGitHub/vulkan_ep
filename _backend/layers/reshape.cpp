#include "Reshape.h"
//cpp stuff
namespace backend {    
   
    Reshape::Reshape() : Layer() { }
       
    vuh::Device* Reshape::_get_device() {
        
        return device;
    }
    
    void Reshape::init() {      
  
    }
    
    void Reshape::bind(std::string _data_input, std::string _shape_input, std::string _reshaped_output){
        data_input = _data_input; shape_input = _shape_input; reshaped_output = _reshaped_output;
		binding.data_input = tensor_dict[data_input]->shape();
  		binding.shape_input = tensor_dict[shape_input]->shape();
 
		binding.reshaped_output = tensor_dict[reshaped_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reshape.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[shape_input]->data(), *tensor_dict[reshaped_output]->data());
    }



}



