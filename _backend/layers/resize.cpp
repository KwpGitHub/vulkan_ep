#include "Resize.h"
//cpp stuff
namespace backend {    
   
    Resize::Resize(std::string name) : Layer(name) { }
       
    vuh::Device* Resize::_get_device() {
        
        return device;
    }
    
    void Resize::init( int _mode) {      
		 mode = _mode; 
  
    }
    
    void Resize::bind(std::string _X_i, std::string _scales_i, std::string _Y_o){
        X_i = _X_i; scales_i = _scales_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
  		binding.scales_i = tensor_dict[scales_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.mode = mode;
 

        
    }
}

