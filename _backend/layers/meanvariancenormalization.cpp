#include "MeanVarianceNormalization.h"
//cpp stuff
namespace backend {    
   
    MeanVarianceNormalization::MeanVarianceNormalization(std::string name) : Layer(name) { }
       
    vuh::Device* MeanVarianceNormalization::_get_device() {
        
        return device;
    }
    
    void MeanVarianceNormalization::init( Shape_t _axes) {      
		 axes = _axes; 
  
    }
    
    void MeanVarianceNormalization::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.axes = axes;
 

        
    }
}

