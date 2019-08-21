#include "Compress.h"
//cpp stuff
namespace backend {    
   
    Compress::Compress(std::string name) : Layer(name) { }
       
    vuh::Device* Compress::_get_device() {
        
        return device;
    }
    
    void Compress::init( int _axis) {      
		 axis = _axis; 
  
    }
    
    void Compress::bind(std::string _input_i, std::string _condition_i, std::string _output_o){
        input_i = _input_i; condition_i = _condition_i; output_o = _output_o;

		binding.input_i = tensor_dict[input_i]->shape();
  		binding.condition_i = tensor_dict[condition_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.axis = axis;
 

        
    }
}

