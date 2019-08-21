#include "Floor.h"
//cpp stuff
namespace backend {    
   
    Floor::Floor(std::string name) : Layer(name) { }
       
    vuh::Device* Floor::_get_device() {
        
        return device;
    }
    
    void Floor::init() {      
  
    }
    
    void Floor::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 


        
    }
}

