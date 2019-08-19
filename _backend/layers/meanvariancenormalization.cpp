#include "MeanVarianceNormalization.h"
//cpp stuff
namespace backend {    
   
    MeanVarianceNormalization::MeanVarianceNormalization() : Layer() { }
       
    vuh::Device* MeanVarianceNormalization::_get_device() {
        
        return device;
    }
    
    void MeanVarianceNormalization::init( Shape_t _axes) {      
		 axes = _axes; 
  
    }
    
    void MeanVarianceNormalization::bind(std::string _X_input, std::string _Y_output){
        X_input = _X_input; Y_output = _Y_output;
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.axes = axes;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/meanvariancenormalization.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }



}



