#include "ConstantOfShape.h"
//cpp stuff
namespace backend {    
   
    ConstantOfShape::ConstantOfShape(const std::string& name) : Layer(name) { }
       
    vuh::Device* ConstantOfShape::_get_device() {
        
        return device;
    }
    
    void ConstantOfShape::init() {      
  
    }
    
    void ConstantOfShape::bind(std::string _value, std::string _input_i, std::string _output_o){
        value = _value; input_i = _input_i; output_o = _output_o;
		binding.input_i = tensor_dict[input_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 

		binding.value = tensor_dict[value]->shape();
 
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/constantofshape.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[value]->data(), *tensor_dict[input_i]->data(), *tensor_dict[output_o]->data());
    }

}

