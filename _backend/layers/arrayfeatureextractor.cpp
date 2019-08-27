#include "arrayfeatureextractor.h"
//cpp stuff
namespace layers {    
   
    ArrayFeatureExtractor::ArrayFeatureExtractor(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/arrayfeatureextractor.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* ArrayFeatureExtractor::_get_device() {        
        return backend::device;
    }
    
    void ArrayFeatureExtractor::init() {      
  
    }
    
    void ArrayFeatureExtractor::bind(std::string _X_i, std::string _Y_i, std::string _Z_o){
        X_i = _X_i; Y_i = _Y_i; Z_o = _Z_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
  		binding.Y_i = backend::tensor_dict[Y_i]->shape();
 
		binding.Z_o = backend::tensor_dict[Z_o]->shape();
 
        
    }

    void ArrayFeatureExtractor::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_i]->data(), *backend::tensor_dict[Z_o]->data());
    }

    void ArrayFeatureExtractor::forward(){ 
        program->run();
    }

}

