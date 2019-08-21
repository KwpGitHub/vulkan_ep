#include "Gather.h"
//cpp stuff
namespace backend {    
   
    Gather::Gather(std::string name) : Layer(name) { }
       
    vuh::Device* Gather::_get_device() {
        
        return device;
    }
    
    void Gather::init( int _axis) {      
		 axis = _axis; 
  
    }
    
    void Gather::bind(std::string _data_i, std::string _indices_i, std::string _output_o){
        data_i = _data_i; indices_i = _indices_i; output_o = _output_o;

		binding.data_i = tensor_dict[data_i]->shape();
  		binding.indices_i = tensor_dict[indices_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.axis = axis;
 

        
    }
}

