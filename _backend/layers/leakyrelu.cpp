#include "leakyrelu.h"
//cpp stuff
namespace layers {    
   
    LeakyRelu::LeakyRelu(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders/bin/leakyrelu.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), file.c_str());
    }
       
    vuh::Device* LeakyRelu::_get_device() {        
        return backend::device;
    }
    
    void LeakyRelu::init( float _alpha) {      
		 alpha = _alpha; 
  
    }
    
    void LeakyRelu::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		binding.X_i = backend::tensor_dict[X_i]->shape();
 
		binding.Y_o = backend::tensor_dict[Y_o]->shape();
 
		//binding.alpha = alpha;
         
    }

    void LeakyRelu::build(){        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        program->bind(binding, *backend::tensor_dict[X_i]->data(), *backend::tensor_dict[Y_o]->data());
    }

    void LeakyRelu::forward(){ 
        //program->run();
    }

}

