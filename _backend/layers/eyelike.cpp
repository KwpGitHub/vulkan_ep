#include "EyeLike.h"
//cpp stuff
namespace backend {    
   
    EyeLike::EyeLike(std::string name) : Layer(name) { }
       
    vuh::Device* EyeLike::_get_device() {
        
        return device;
    }
    
    void EyeLike::init( int _dtype,  int _k) {      
		 dtype = _dtype; 
 		 k = _k; 
  
    }
    
    void EyeLike::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		binding.input_i = tensor_dict[input_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.dtype = dtype;
  		binding.k = k;
 

        
    }
}

