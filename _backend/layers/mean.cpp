#include "Mean.h"
//cpp stuff
namespace backend {    
   
    Mean::Mean(std::string name) : Layer(name) { }
       
    vuh::Device* Mean::_get_device() {
        
        return device;
    }
    
    void Mean::init() {      
  
    }
    
    void Mean::bind(std::string _mean_o){
        mean_o = _mean_o;


		binding.mean_o = tensor_dict[mean_o]->shape();
 


        
    }
}

