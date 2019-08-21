#include "Tanh.h"
//cpp stuff
namespace backend {    
   
    Tanh::Tanh(std::string name) : Layer(name) { }
       
    vuh::Device* Tanh::_get_device() {
        
        return device;
    }
    
    void Tanh::init() {      
  
    }
    
    void Tanh::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		binding.input_i = tensor_dict[input_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 


        
    }
}

