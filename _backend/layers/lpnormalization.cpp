#include "LpNormalization.h"
//cpp stuff
namespace backend {    
   
    LpNormalization::LpNormalization(std::string name) : Layer(name) { }
       
    vuh::Device* LpNormalization::_get_device() {
        
        return device;
    }
    
    void LpNormalization::init( int _axis,  int _p) {      
		 axis = _axis; 
 		 p = _p; 
  
    }
    
    void LpNormalization::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		binding.input_i = tensor_dict[input_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.axis = axis;
  		binding.p = p;
 

        
    }
}

