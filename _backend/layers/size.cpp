#include "Size.h"
//cpp stuff
namespace backend {    
   
    Size::Size(std::string name) : Layer(name) { }
       
    vuh::Device* Size::_get_device() {
        
        return device;
    }
    
    void Size::init() {      
  
    }
    
    void Size::bind(std::string _data_i, std::string _size_o){
        data_i = _data_i; size_o = _size_o;

		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.size_o = tensor_dict[size_o]->shape();
 


        
    }
}

