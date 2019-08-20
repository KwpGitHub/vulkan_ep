#include "Mean.h"
//cpp stuff
namespace backend {    
   
    Mean::Mean(const std::string& name) : Layer(name) { }
       
    vuh::Device* Mean::_get_device() {
        
        return device;
    }
    
    void Mean::init() {      
  
    }
    
    void Mean::bind(std::string _mean_output){
        mean_output = _mean_output;

		binding.mean_output = tensor_dict[mean_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/mean.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[mean_output]->data());
    }

}

