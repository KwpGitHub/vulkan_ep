#include "pow.h"
//cpp stuff
namespace layers {    
   
    Pow::Pow(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/pow.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Pow::_get_device() {        
        return backend::device;
    }
    
    void Pow::init() {      
  
    }
    
    void Pow::bind(std::string _X_i, std::string _Y_i, std::string _Z_o){
        X_i = _X_i; Y_i = _Y_i; Z_o = _Z_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
  		binding.Y_i = backend::tensor_dict[Y_i]->shape();
 
		binding.Z_o = backend::tensor_dict[Z_o]->shape();
 
        
    }

    void Pow::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_i]->data(), *backend::tensor_dict[Z_o]->data());
    }

    void Pow::forward(){ 
        program->run();
    }

}

