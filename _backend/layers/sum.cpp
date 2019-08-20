#include "Sum.h"
//cpp stuff
namespace backend {    
   
    Sum::Sum(const std::string& name) : Layer(name) { }
       
    vuh::Device* Sum::_get_device() {
        
        return device;
    }
    
    void Sum::init() {      
  
    }
    
    void Sum::bind(std::string _sum_output){
        sum_output = _sum_output;

		binding.sum_output = tensor_dict[sum_output]->shape();
 


        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/sum.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[sum_output]->data());
    }

}

