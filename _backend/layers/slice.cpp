#include "Slice.h"
//cpp stuff
namespace backend {    
   
    Slice::Slice(std::string name) : Layer(name) { }
       
    vuh::Device* Slice::_get_device() {
        
        return device;
    }
    
    void Slice::init() {      
  
    }
    
    void Slice::bind(std::string _data_i, std::string _starts_i, std::string _ends_i, std::string _axes_i, std::string _steps_i, std::string _output_o){
        data_i = _data_i; starts_i = _starts_i; ends_i = _ends_i; axes_i = _axes_i; steps_i = _steps_i; output_o = _output_o;

		binding.data_i = tensor_dict[data_i]->shape();
  		binding.starts_i = tensor_dict[starts_i]->shape();
  		binding.ends_i = tensor_dict[ends_i]->shape();
  		binding.axes_i = tensor_dict[axes_i]->shape();
  		binding.steps_i = tensor_dict[steps_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 


        
    }
}

