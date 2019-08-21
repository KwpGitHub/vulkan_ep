#include "Max.h"
//cpp stuff
namespace backend {    
   
    Max::Max(std::string name) : Layer(name) { }
       
    vuh::Device* Max::_get_device() {
        
        return device;
    }
    
    void Max::init() {      
  
    }
    
    void Max::bind(std::string _max_o){
        max_o = _max_o;


		binding.max_o = tensor_dict[max_o]->shape();
 


        
    }
}

