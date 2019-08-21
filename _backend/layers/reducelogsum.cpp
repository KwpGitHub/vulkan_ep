#include "ReduceLogSum.h"
//cpp stuff
namespace backend {    
   
    ReduceLogSum::ReduceLogSum(std::string name) : Layer(name) { }
       
    vuh::Device* ReduceLogSum::_get_device() {
        
        return device;
    }
    
    void ReduceLogSum::init( Shape_t _axes,  int _keepdims) {      
		 axes = _axes; 
 		 keepdims = _keepdims; 
  
    }
    
    void ReduceLogSum::bind(std::string _data_i, std::string _reduced_o){
        data_i = _data_i; reduced_o = _reduced_o;

		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.reduced_o = tensor_dict[reduced_o]->shape();
 
		binding.axes = axes;
  		binding.keepdims = keepdims;
 

        
    }
}

