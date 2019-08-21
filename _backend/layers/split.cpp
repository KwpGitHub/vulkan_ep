#include "Split.h"
//cpp stuff
namespace backend {    
   
    Split::Split(std::string name) : Layer(name) { }
       
    vuh::Device* Split::_get_device() {
        
        return device;
    }
    
    void Split::init( int _axis,  Shape_t _split) {      
		 axis = _axis; 
 		 split = _split; 
  
    }
    
    void Split::bind(std::string _input_i){
        input_i = _input_i;

		binding.input_i = tensor_dict[input_i]->shape();
 

		binding.axis = axis;
  		binding.split = split;
 

        
    }
}

