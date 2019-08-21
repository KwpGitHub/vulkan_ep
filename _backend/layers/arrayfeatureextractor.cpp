#include "ArrayFeatureExtractor.h"
//cpp stuff
namespace backend {    
   
    ArrayFeatureExtractor::ArrayFeatureExtractor(std::string name) : Layer(name) { }
       
    vuh::Device* ArrayFeatureExtractor::_get_device() {
        
        return device;
    }
    
    void ArrayFeatureExtractor::init() {      
  
    }
    
    void ArrayFeatureExtractor::bind(std::string _X_i, std::string _Y_i, std::string _Z_o){
        X_i = _X_i; Y_i = _Y_i; Z_o = _Z_o;

		binding.X_i = tensor_dict[X_i]->shape();
  		binding.Y_i = tensor_dict[Y_i]->shape();
 
		binding.Z_o = tensor_dict[Z_o]->shape();
 


        
    }
}

