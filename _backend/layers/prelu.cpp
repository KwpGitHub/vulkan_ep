#include "prelu.h"
//cpp stuff
namespace layers {    
   
    PRelu::PRelu(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/prelu.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* PRelu::_get_device() {        
        return backend::device;
    }
    
    void PRelu::init() {      
  
    }
    
    void PRelu::bind(std::string _X_i, std::string _slope_i, std::string _Y_o){
        X_i = _X_i; slope_i = _slope_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
  		binding.slope_i = backend::tensor_dict[slope_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
        
    }

    void PRelu::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[slope_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void PRelu::forward(){ 
        //program->run();
    }

}

