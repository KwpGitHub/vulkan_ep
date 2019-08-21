#include "FeatureVectorizer.h"
//cpp stuff
namespace backend {    
   
    FeatureVectorizer::FeatureVectorizer(std::string name) : Layer(name) { }
       
    vuh::Device* FeatureVectorizer::_get_device() {
        
        return device;
    }
    
    void FeatureVectorizer::init( Shape_t _inputdimensions) {      
		 inputdimensions = _inputdimensions; 
  
    }
    
    void FeatureVectorizer::bind(std::string _Y_o){
        Y_o = _Y_o;


		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.inputdimensions = inputdimensions;
 

        
    }
}

