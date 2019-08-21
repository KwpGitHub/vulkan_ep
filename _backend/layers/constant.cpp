#include "Constant.h"
//cpp stuff
namespace backend {    
   
    Constant::Constant(std::string name) : Layer(name) { }
       
    vuh::Device* Constant::_get_device() {
        
        return device;
    }
    
    void Constant::init() {      
  
    }
    
    void Constant::bind(std::string _value, std::string _output_o){
        value = _value; output_o = _output_o;


		binding.output_o = tensor_dict[output_o]->shape();
 

		binding.value = tensor_dict[value]->shape();
 
        
    }
}

