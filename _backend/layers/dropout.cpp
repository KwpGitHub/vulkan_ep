#include "Dropout.h"
//cpp stuff
namespace backend {    
   
    Dropout::Dropout(std::string name) : Layer(name) { }
       
    vuh::Device* Dropout::_get_device() {
        
        return device;
    }
    
    void Dropout::init( float _ratio) {      
		 ratio = _ratio; 
  
    }
    
    void Dropout::bind(std::string _data_i, std::string _output_o, std::string _mask_o){
        data_i = _data_i; output_o = _output_o; mask_o = _mask_o;

		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
  		binding.mask_o = tensor_dict[mask_o]->shape();
 
		binding.ratio = ratio;
 

        
    }
}

