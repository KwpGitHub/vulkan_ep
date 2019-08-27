#include "resize.h"
//cpp stuff
namespace layers {    
   
    Resize::Resize(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/resize.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* Resize::_get_device() {        
        return backend::device;
    }
    
    void Resize::init( std::string _mode) {      
		 mode = _mode; 
  
    }
    
    void Resize::bind(std::string _X_i, std::string _scales_i, std::string _Y_o){
        X_i = _X_i; scales_i = _scales_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
  		binding.scales_i = backend::tensor_dict[scales_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.mode = mode;
         
    }

    void Resize::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[scales_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void Resize::forward(){ 
        program->run();
    }

}

