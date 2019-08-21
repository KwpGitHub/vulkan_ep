#include "Min.h"
//cpp stuff
namespace backend {    
   
    Min::Min(std::string name) : Layer(name) { }
       
    vuh::Device* Min::_get_device() {
        
        return device;
    }
    
    void Min::init() {      
  
    }
    
    void Min::bind(std::string _min_o){
        min_o = _min_o;


		binding.min_o = tensor_dict[min_o]->shape();
 


        
    }
}

