#include "ReduceMin.h"
//cpp stuff
namespace backend {    
   
    ReduceMin::ReduceMin(const std::string& name) : Layer(name) { }
       
    vuh::Device* ReduceMin::_get_device() {
        
        return device;
    }
    
    void ReduceMin::init( Shape_t _axes,  int _keepdims) {      
		 axes = _axes; 
 		 keepdims = _keepdims; 
  
    }
    
    void ReduceMin::bind(std::string _data_i, std::string _reduced_o){
        data_i = _data_i; reduced_o = _reduced_o;
		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.reduced_o = tensor_dict[reduced_o]->shape();
 
		binding.axes = axes;
  		binding.keepdims = keepdims;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reducemin.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[reduced_o]->data());
    }

}

