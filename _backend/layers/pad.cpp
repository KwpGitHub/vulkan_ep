#include "Pad.h"
//cpp stuff
namespace backend {    
   
    Pad::Pad(std::string name) : Layer(name) { }
       
    vuh::Device* Pad::_get_device() {
        
        return device;
    }
    
    void Pad::init( Shape_t _pads,  int _mode,  float _value) {      
		 pads = _pads; 
 		 mode = _mode; 
 		 value = _value; 
  
    }
    
    void Pad::bind(std::string _data_i, std::string _output_o){
        data_i = _data_i; output_o = _output_o;

		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.pads = pads;
  		binding.mode = mode;
  		binding.value = value;
 

        
    }
}

