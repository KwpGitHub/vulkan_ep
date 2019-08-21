#include "Squeeze.h"
//cpp stuff
namespace backend {    
   
    Squeeze::Squeeze(std::string name) : Layer(name) { }
       
    vuh::Device* Squeeze::_get_device() {
        
        return device;
    }
    
    void Squeeze::init( Shape_t _axes) {      
		 axes = _axes; 
  
    }
    
    void Squeeze::bind(std::string _data_i, std::string _squeezed_o){
        data_i = _data_i; squeezed_o = _squeezed_o;

		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.squeezed_o = tensor_dict[squeezed_o]->shape();
 
		binding.axes = axes;
 

        
    }
}

