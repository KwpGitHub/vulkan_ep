#include "Where.h"
//cpp stuff
namespace backend {    
   
    Where::Where(std::string name) : Layer(name) { }
       
    vuh::Device* Where::_get_device() {
        
        return device;
    }
    
    void Where::init() {      
  
    }
    
    void Where::bind(std::string _condition_i, std::string _X_i, std::string _Y_i, std::string _output_o){
        condition_i = _condition_i; X_i = _X_i; Y_i = _Y_i; output_o = _output_o;

		binding.condition_i = tensor_dict[condition_i]->shape();
  		binding.X_i = tensor_dict[X_i]->shape();
  		binding.Y_i = tensor_dict[Y_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 


        
    }
}

