#include "isinf.h"
//cpp stuff
namespace layers {    
   
    IsInf::IsInf(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/isinf.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* IsInf::_get_device() {        
        return backend::device;
    }
    
    void IsInf::init( int _detect_negative,  int _detect_positive) {      
		 detect_negative = _detect_negative; 
 		 detect_positive = _detect_positive; 
  
    }
    
    void IsInf::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.detect_negative = detect_negative;
  		//binding.detect_positive = detect_positive;
         
    }

    void IsInf::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void IsInf::forward(){ 
        program->run();
    }

}

