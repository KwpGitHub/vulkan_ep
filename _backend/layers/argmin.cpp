#include "ArgMin.h"
//cpp stuff
namespace backend {    
   
    ArgMin::ArgMin(std::string name) : Layer(name) { }
       
    vuh::Device* ArgMin::_get_device() {
        
        return device;
    }
    
    void ArgMin::init( int _axis,  int _keepdims) {      
		 axis = _axis; 
 		 keepdims = _keepdims; 
  
    }
    
    void ArgMin::bind(std::string _data_i, std::string _reduced_o){
        data_i = _data_i; reduced_o = _reduced_o;

		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.reduced_o = tensor_dict[reduced_o]->shape();
 
		binding.axis = axis;
  		binding.keepdims = keepdims;
 

        
    }
}

