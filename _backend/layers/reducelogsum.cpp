#include "ReduceLogSum.h"
//cpp stuff
namespace backend {    
   
    ReduceLogSum::ReduceLogSum() : Layer() { }
       
    vuh::Device* ReduceLogSum::_get_device() {
        
        return device;
    }
    
    void ReduceLogSum::init( Shape_t _axes,  int _keepdims) {      
		 axes = _axes; 
 		 keepdims = _keepdims; 
  
    }
    
    void ReduceLogSum::bind(std::string _data_input, std::string _reduced_output){
        data_input = _data_input; reduced_output = _reduced_output;
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.reduced_output = tensor_dict[reduced_output]->shape();
 
		binding.axes = axes;
  		binding.keepdims = keepdims;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/reducelogsum.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[reduced_output]->data());
    }



}



