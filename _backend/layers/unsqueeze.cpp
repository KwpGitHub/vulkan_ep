#include "Unsqueeze.h"
//cpp stuff
namespace backend {    
   
    Unsqueeze::Unsqueeze(std::string name) : Layer(name) { }
       
    vuh::Device* Unsqueeze::_get_device() {
        
        return device;
    }
    
    void Unsqueeze::init( Shape_t _axes) {      
		 axes = _axes; 
  
    }
    
    void Unsqueeze::bind(std::string _data_i, std::string _expanded_o){
        data_i = _data_i; expanded_o = _expanded_o;

		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.expanded_o = tensor_dict[expanded_o]->shape();
 
		binding.axes = axes;
 

        
    }
}

