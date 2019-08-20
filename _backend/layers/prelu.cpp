#include "PRelu.h"
//cpp stuff
namespace backend {    
   
    PRelu::PRelu(const std::string& name) : Layer(name) { }
       
    vuh::Device* PRelu::_get_device() {
        
        return device;
    }
    
    void PRelu::init() {      
  
    }
    
    void PRelu::bind(std::string _X_input, std::string _slope_input, std::string _Y_output){
        X_input = _X_input; slope_input = _slope_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.slope_input = tensor_dict[slope_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/prelu.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[slope_input]->data(), *tensor_dict[Y_output]->data());
    }

}

