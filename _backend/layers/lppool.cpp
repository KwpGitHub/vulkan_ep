#include "LpPool.h"
//cpp stuff
namespace backend {    
   
    LpPool::LpPool(std::string name) : Layer(name) { }
       
    vuh::Device* LpPool::_get_device() {
        
        return device;
    }
    
    void LpPool::init( Shape_t _kernel_shape,  int _auto_pad,  int _p,  Shape_t _pads,  Shape_t _strides) {      
		 kernel_shape = _kernel_shape; 
 		 auto_pad = _auto_pad; 
 		 p = _p; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void LpPool::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.kernel_shape = kernel_shape;
  		binding.auto_pad = auto_pad;
  		binding.p = p;
  		binding.pads = pads;
  		binding.strides = strides;
 

        
    }
}

