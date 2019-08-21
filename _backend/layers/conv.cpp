#include "Conv.h"
//cpp stuff
namespace backend {    
   
    Conv::Conv(std::string name) : Layer(name) { }
       
    vuh::Device* Conv::_get_device() {
        
        return device;
    }
    
    void Conv::init( int _auto_pad,  Shape_t _dilations,  int _group,  Shape_t _kernel_shape,  Shape_t _pads,  Shape_t _strides) {      
		 auto_pad = _auto_pad; 
 		 dilations = _dilations; 
 		 group = _group; 
 		 kernel_shape = _kernel_shape; 
 		 pads = _pads; 
 		 strides = _strides; 
  
    }
    
    void Conv::bind(std::string _X_i, std::string _W_i, std::string _B_i, std::string _Y_o){
        X_i = _X_i; W_i = _W_i; B_i = _B_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
  		binding.W_i = tensor_dict[W_i]->shape();
  		binding.B_i = tensor_dict[B_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.auto_pad = auto_pad;
  		binding.dilations = dilations;
  		binding.group = group;
  		binding.kernel_shape = kernel_shape;
  		binding.pads = pads;
  		binding.strides = strides;
 

        
    }
}

