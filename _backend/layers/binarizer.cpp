#include "binarizer.h"
//cpp stuff
namespace layers {    
   
    Binarizer::Binarizer(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/binarizer.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Binarizer::_get_device() {        
        return backend::device;
    }
    
    void Binarizer::init( float _threshold) {      
		 threshold = _threshold; 
  
    }
    
    void Binarizer::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.threshold = threshold;
         
    }

    void Binarizer::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void Binarizer::forward(){ 
        //program->run();
    }

}

