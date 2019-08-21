#include "TopK.h"
//cpp stuff
namespace backend {    
   
    TopK::TopK(std::string name) : Layer(name) { }
       
    vuh::Device* TopK::_get_device() {
        
        return device;
    }
    
    void TopK::init( int _axis) {      
		 axis = _axis; 
  
    }
    
    void TopK::bind(std::string _X_i, std::string _K_i, std::string _Values_o, std::string _Indices_o){
        X_i = _X_i; K_i = _K_i; Values_o = _Values_o; Indices_o = _Indices_o;

		binding.X_i = tensor_dict[X_i]->shape();
  		binding.K_i = tensor_dict[K_i]->shape();
 
		binding.Values_o = tensor_dict[Values_o]->shape();
  		binding.Indices_o = tensor_dict[Indices_o]->shape();
 
		binding.axis = axis;
 

        
    }
}

