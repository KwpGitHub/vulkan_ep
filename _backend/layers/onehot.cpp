#include "OneHot.h"
//cpp stuff
namespace backend {    
   
    OneHot::OneHot(std::string name) : Layer(name) { }
       
    vuh::Device* OneHot::_get_device() {
        
        return device;
    }
    
    void OneHot::init( int _axis) {      
		 axis = _axis; 
  
    }
    
    void OneHot::bind(std::string _indices_i, std::string _depth_i, std::string _values_i, std::string _output_o){
        indices_i = _indices_i; depth_i = _depth_i; values_i = _values_i; output_o = _output_o;

		binding.indices_i = tensor_dict[indices_i]->shape();
  		binding.depth_i = tensor_dict[depth_i]->shape();
  		binding.values_i = tensor_dict[values_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.axis = axis;
 

        
    }
}

