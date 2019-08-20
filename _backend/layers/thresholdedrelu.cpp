#include "ThresholdedRelu.h"
//cpp stuff
namespace backend {    
   
    ThresholdedRelu::ThresholdedRelu(const std::string& name) : Layer(name) { }
       
    vuh::Device* ThresholdedRelu::_get_device() {
        
        return device;
    }
    
    void ThresholdedRelu::init( float _alpha) {      
		 alpha = _alpha; 
  
    }
    
    void ThresholdedRelu::bind(std::string _X_input, std::string _Y_output){
        X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.alpha = alpha;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/thresholdedrelu.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }

}

