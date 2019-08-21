#include "QuantizeLinear.h"
//cpp stuff
namespace backend {    
   
    QuantizeLinear::QuantizeLinear(std::string name) : Layer(name) { }
       
    vuh::Device* QuantizeLinear::_get_device() {
        
        return device;
    }
    
    void QuantizeLinear::init() {      
  
    }
    
    void QuantizeLinear::bind(std::string _x_i, std::string _y_scale_i, std::string _y_zero_point_i, std::string _y_o){
        x_i = _x_i; y_scale_i = _y_scale_i; y_zero_point_i = _y_zero_point_i; y_o = _y_o;

		binding.x_i = tensor_dict[x_i]->shape();
  		binding.y_scale_i = tensor_dict[y_scale_i]->shape();
  		binding.y_zero_point_i = tensor_dict[y_zero_point_i]->shape();
 
		binding.y_o = tensor_dict[y_o]->shape();
 


        
    }
}

