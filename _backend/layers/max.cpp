#include "Max.h"
//cpp stuff
namespace backend {    
   
    Max::Max() : Layer() { }
       
    vuh::Device* Max::_get_device() {
        
        return device;
    }
    
    void Max::init() {      
  
    }
    
    void Max::bind(std::string _max_output){
        max_output = _max_output;

		binding.max_output = tensor_dict[max_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/max.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[max_output]->data());
    }



}



