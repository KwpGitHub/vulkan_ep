#include "ArgMax.h"
//cpp stuff
namespace backend {    
   
    ArgMax::ArgMax(const std::string& name) : Layer(name) { }
       
    vuh::Device* ArgMax::_get_device() {
        
        return device;
    }
    
    void ArgMax::init( int _axis,  int _keepdims) {      
		 axis = _axis; 
 		 keepdims = _keepdims; 
  
    }
    
    void ArgMax::bind(std::string _data_i, std::string _reduced_o){
        data_i = _data_i; reduced_o = _reduced_o;
		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.reduced_o = tensor_dict[reduced_o]->shape();
 
		binding.axis = axis;
  		binding.keepdims = keepdims;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/argmax.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[reduced_o]->data());
    }

}

