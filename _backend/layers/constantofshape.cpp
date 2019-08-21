#include "ConstantOfShape.h"
//cpp stuff
namespace backend {    
   
    ConstantOfShape::ConstantOfShape(std::string name) : Layer(name) { }
       
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
 
        
    }
}

