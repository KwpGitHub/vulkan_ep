#include "Mean.h"
//cpp stuff
namespace backend {    
   
    Mean::Mean(const std::string& name) : Layer(name) { }
       
    vuh::Device* Mean::_get_device() {
        
        return device;
    }
    
    void Mean::init() {      
  
    }
    
    void Mean::bind(std::string _mean_o){
        mean_o = _mean_o;

		binding.mean_o = tensor_dict[mean_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/mean.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[mean_o]->data());
    }

}

