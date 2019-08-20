#include "ArgMin.h"
//cpp stuff
namespace backend {    
   
    ArgMin::ArgMin(const std::string& name) : Layer(name) { }
       
    vuh::Device* ArgMin::_get_device() {
        
        return device;
    }
    
    void ArgMin::init( int _axis,  int _keepdims) {      
		 axis = _axis; 
 		 keepdims = _keepdims; 
  
    }
    
    void ArgMin::bind(std::string _data_i, std::string _reduced_o){
        data_i = _data_i; reduced_o = _reduced_o;
		binding.data_i = tensor_dict[data_i]->shape();
 
		binding.reduced_o = tensor_dict[reduced_o]->shape();
 
		binding.axis = axis;
  		binding.keepdims = keepdims;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/argmin.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[data_i]->data(), *tensor_dict[reduced_o]->data());
    }

}

