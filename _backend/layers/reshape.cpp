#include "Reshape.h"
//cpp stuff
namespace backend {    
   
    Reshape::Reshape(const std::string& name) : Layer(name) { }
       
    vuh::Device* Reshape::_get_device() {
        
        return device;
    }
    
    void Reshape::init() {      
  
    }
    
    void Reshape::bind(std::string _data_i, std::string _shape_i, std::string _reshaped_o){
        data_i = _data_i; shape_i = _shape_i; reshaped_o = _reshaped_o;
		binding.data_i = tensor_dict[data_i]->shape();
  		binding.shape_i = tensor_dict[shape_i]->shape();
 
		binding.reshaped_o = tensor_dict[reshaped_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reshape.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[shape_i]->data(), *tensor_dict[reshaped_o]->data());
    }

}

