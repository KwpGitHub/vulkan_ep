#include "Shrink.h"
//cpp stuff
namespace backend {    
   
    Shrink::Shrink(std::string name) : Layer(name) { }
       
    vuh::Device* Shrink::_get_device() {
        
        return device;
    }
    
    void Shrink::init( float _bias,  float _lambd) {      
		 bias = _bias; 
 		 lambd = _lambd; 
  
    }
    
    void Shrink::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		binding.input_i = tensor_dict[input_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.bias = bias;
  		binding.lambd = lambd;
 

        
    }
}

