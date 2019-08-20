#include "Expand.h"
//cpp stuff
namespace backend {    
   
    Expand::Expand(const std::string& name) : Layer(name) { }
       
    vuh::Device* Expand::_get_device() {
        
        return device;
    }
    
    void Expand::init() {      
  
    }
    
    void Expand::bind(std::string _input_i, std::string _shape_i, std::string _output_o){
        input_i = _input_i; shape_i = _shape_i; output_o = _output_o;
		binding.input_i = tensor_dict[input_i]->shape();
  		binding.shape_i = tensor_dict[shape_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/expand.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[shape_i]->data(), *tensor_dict[output_o]->data());
    }

}
