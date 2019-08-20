#include "Constant.h"
//cpp stuff
namespace backend {    
   
    Constant::Constant(const std::string& name) : Layer(name) { }
       
    vuh::Device* Constant::_get_device() {
        
        return device;
    }
    
    void Constant::init() {      
  
    }
    
    void Constant::bind(std::string _value, std::string _output_o){
        value = _value; output_o = _output_o;

		binding.output_o = tensor_dict[output_o]->shape();
 

		binding.value = tensor_dict[value]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/constant.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[value]->data(), *tensor_dict[output_o]->data());
    }

}

