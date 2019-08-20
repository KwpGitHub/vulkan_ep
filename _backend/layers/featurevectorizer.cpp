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
    
    void FeatureVectorizer::bind(std::string _Y_output){
        Y_output = _Y_output;

		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.inputdimensions = inputdimensions;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/featurevectorizer.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[Y_output]->data());
    }

}

