#include "ReduceL2.h"
//cpp stuff
namespace backend {    
   
    ReduceL2::ReduceL2(std::string name) : Layer(name) { }
       
    vuh::Device* ReduceL2::_get_device() {
        
        return device;
    }
    
    void ReduceL2::init( Shape_t _axes,  int _keepdims) {      
		 axes = _axes; 
 		 keepdims = _keepdims; 
  
    }
    
    void ReduceL2::bind(std::string _data_i, std::string _reduced_o){
        data_i = _data_i; reduced_o = _reduced_o;

		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.reduced_o = tensor_dict[reduced_o]->shape();
 
		binding.axes = axes;
  		binding.keepdims = keepdims;
 

        
    }
}

