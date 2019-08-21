#include "ReduceMin.h"
//cpp stuff
namespace backend {    
   
    ReduceMin::ReduceMin(std::string name) : Layer(name) { }
       
    vuh::Device* ReduceMin::_get_device() {
        
        return device;
    }
    
    void ReduceMin::init( Shape_t _axes,  int _keepdims) {      
		 axes = _axes; 
 		 keepdims = _keepdims; 
  
    }
    
    void ReduceMin::bind(std::string _data_i, std::string _reduced_o){
        data_i = _data_i; reduced_o = _reduced_o;

		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.reduced_o = tensor_dict[reduced_o]->shape();
 
		binding.axes = axes;
  		binding.keepdims = keepdims;
 

        
    }
}

