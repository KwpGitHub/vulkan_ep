#include "Min.h"
//cpp stuff
namespace backend {    
   
    Min::Min() : Layer() { }
       
    vuh::Device* Min::_get_device() {
        
        return device;
    }
    
    void Min::init() {      
  
    }
    
    void Min::bind(std::string _min_output){
        min_output = _min_output;

		binding.min_output = tensor_dict[min_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/min.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[min_output]->data());
    }



}



