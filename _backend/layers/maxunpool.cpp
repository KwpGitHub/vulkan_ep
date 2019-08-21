#include "MaxUnpool.h"
//cpp stuff
namespace backend {    
   
    MaxUnpool::MaxUnpool(std::string name) : Layer(name) { }
       
    vuh::Device* MaxUnpool::_get_device() {
        
        return device;
    }
    
    void MaxUnpool::init( Shape_t _kernel_shape,  Shape_t _pads,  Shape_t _strides) {      
		 kernel_shape = _kernel_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void MaxUnpool::bind(std::string _X_i, std::string _I_i, std::string _output_shape_i, std::string _output_o){
        X_i = _X_i; I_i = _I_i; output_shape_i = _output_shape_i; output_o = _output_o;

		binding.X_i = tensor_dict[X_i]->shape();
  		binding.I_i = tensor_dict[I_i]->shape();
  		binding.output_shape_i = tensor_dict[output_shape_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.kernel_shape = kernel_shape;
  		binding.pads = pads;
  		binding.strides = strides;
 

        
    }
}

