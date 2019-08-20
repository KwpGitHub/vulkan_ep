#include "OneHot.h"
//cpp stuff
namespace backend {    
   
    OneHot::OneHot(const std::string& name) : Layer(name) { }
       
    vuh::Device* OneHot::_get_device() {
        
        return device;
    }
    
    void OneHot::init( int _axis) {      
		 axis = _axis; 
  
    }
    
    void OneHot::bind(std::string _indices_i, std::string _depth_i, std::string _values_i, std::string _output_o){
        indices_i = _indices_i; depth_i = _depth_i; values_i = _values_i; output_o = _output_o;
		binding.indices_i = tensor_dict[indices_i]->shape();
  		binding.depth_i = tensor_dict[depth_i]->shape();
  		binding.values_i = tensor_dict[values_i]->shape();
 
		binding.output_o = tensor_dict[output_o]->shape();
 
		binding.axis = axis;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/onehot.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[indices_i]->data(), *tensor_dict[depth_i]->data(), *tensor_dict[values_i]->data(), *tensor_dict[output_o]->data());
    }

}

