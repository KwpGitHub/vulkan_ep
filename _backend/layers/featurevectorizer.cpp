#include "featurevectorizer.h"
//cpp stuff
namespace layers {    
   
    FeatureVectorizer::FeatureVectorizer(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\featurevectorizer.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* FeatureVectorizer::_get_device() {
        
        return backend::device;
    }
    
    void FeatureVectorizer::init( std::vector<int> _inputdimensions) {      
		 inputdimensions = _inputdimensions; 
  
    }
    
    void FeatureVectorizer::bind(std::string _Y_o){
        Y_o = _Y_o;


		//binding.Y_o = tensor_dict[Y_o]->shape();
 
		//binding.inputdimensions = inputdimensions;
         
    }

    void FeatureVectorizer::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[Y_o]->data());
    }

}

