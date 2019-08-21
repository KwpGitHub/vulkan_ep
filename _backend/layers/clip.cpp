#include "Clip.h"
//cpp stuff
namespace backend {    
   
    Clip::Clip(std::string name) : Layer(name) { }
       
    vuh::Device* Clip::_get_device() {
        
        return device;
    }
    
    void Clip::init( float _max,  float _min) {      
		 max = _max; 
 		 min = _min; 
  
    }
    
    void Clip::bind(std::string _input_i, std::string _output_o){
        input_i = _input_i; output_o = _output_o;

		binding.input_i = tensor_dict[input_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.max = max;
  		binding.min = min;
 

        
    }
}

