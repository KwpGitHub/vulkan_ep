#include "Selu.h"
//cpp stuff
namespace backend {    
   
    Selu::Selu(const std::string& name) : Layer(name) { }
       
    vuh::Device* Selu::_get_device() {
        
        return device;
    }
    
    void Selu::init( float _alpha,  float _gamma) {      
		 alpha = _alpha; 
 		 gamma = _gamma; 
  
    }
    
    void Selu::bind(std::string _X_input, std::string _Y_output){
        X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.alpha = alpha;
  		binding.gamma = gamma;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/selu.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }

}

