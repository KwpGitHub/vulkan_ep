#include "Elu.h"
//cpp stuff
namespace backend {    
   
    Elu::Elu(const std::string& name) : Layer(name) { }
       
    vuh::Device* Elu::_get_device() {
        
        return device;
    }
    
    void Elu::init( float _alpha) {      
		 alpha = _alpha; 
  
    }
    
    void Elu::bind(std::string _X_i, std::string _Y_o){
        X_i = _X_i; Y_o = _Y_o;
		binding.X_i = tensor_dict[X_i]->shape();
 
		binding.Y_o = tensor_dict[Y_o]->shape();
 
		binding.alpha = alpha;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/elu.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
    }

}

