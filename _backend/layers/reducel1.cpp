#include "ReduceL1.h"
//cpp stuff
namespace backend {    
   
    ReduceL1::ReduceL1(std::string name) : Layer(name) { }
       
    vuh::Device* ReduceL1::_get_device() {
        
        return device;
    }
    
    void ReduceL1::init( Shape_t _axes,  int _keepdims) {      
		 axes = _axes; 
 		 keepdims = _keepdims; 
  
    }
    
    void ReduceL1::bind(std::string _data_i, std::string _reduced_o){
        data_i = _data_i; reduced_o = _reduced_o;

		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.reduced_o = tensor_dict[reduced_o]->shape();
 
		binding.axes = axes;
  		binding.keepdims = keepdims;
 

        
    }
}

