#include "Sin.h"
//cpp stuff
namespace backend {    
   
    Sin::Sin(std::string name) : Layer(name) { }
       
    vuh::Device* Sin::_get_device() {
        
        return device;
    }
    
    void Sin::init() {      
  
    }
    
    void Sin::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		binding.input_i = tensor_dict[input_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 


        
    }
}

