#include "FeatureVectorizer.h"
//cpp stuff
namespace backend {    
   
    FeatureVectorizer::FeatureVectorizer(const std::string& name) : Layer(name) { }
       
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
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/featurevectorizer.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[Y_o]->data());
    }

}

