#include "Min.h"
//cpp stuff
namespace backend {    
   
    Min::Min(const std::string& name) : Layer(name) { }
       
    vuh::Device* Min::_get_device() {
        
        return device;
    }
    
    void Min::init() {      
  
    }
    
    void Min::bind(std::string _min_o){
        min_o = _min_o;

		binding.min_o = tensor_dict[min_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/min.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[min_o]->data());
    }

}

