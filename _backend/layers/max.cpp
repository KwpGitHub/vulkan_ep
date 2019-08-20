#include "Max.h"
//cpp stuff
namespace backend {    
   
    Max::Max(const std::string& name) : Layer(name) { }
       
    vuh::Device* Max::_get_device() {
        
        return device;
    }
    
    void Max::init() {      
  
    }
    
    void Max::bind(std::string _max_o){
        max_o = _max_o;

		binding.max_o = tensor_dict[max_o]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/max.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[max_o]->data());
    }

}

