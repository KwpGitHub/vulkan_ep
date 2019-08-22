#include "selu.h"
//cpp stuff
namespace layers {    
   
    Selu::Selu(std::string name) : backend::Layer(name) {    
        std::string file;
        file.append(backend::file_path);
        file.append("shaders\\bin\\selu.spv");
        program = new vuh::Program<Specs, binding_descriptor>(*backend::device, file.c_str());
    }
       
    vuh::Device* Selu::_get_device() {
        
        return backend::device;
    }
    
    void Selu::init( float _alpha,  float _gamma) {      
		 alpha = _alpha; 
 		 gamma = _gamma; 
  
    }
    
    void Selu::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;

		//binding.X_i = tensor_dict[X_i]->shape();
 
		//binding.Y_o = tensor_dict[Y_o]->shape();
 
		//binding.alpha = alpha;
  		//binding.gamma = gamma;
         
    }

    void Selu::build(){
        
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE).spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

