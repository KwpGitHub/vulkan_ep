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
    
    void OneHot::bind(std::string _indices_input, std::string _depth_input, std::string _values_input, std::string _output_output){
        indices_input = _indices_input; depth_input = _depth_input; values_input = _values_input; output_output = _output_output;
		binding.indices_input = tensor_dict[indices_input]->shape();
  		binding.depth_input = tensor_dict[depth_input]->shape();
  		binding.values_input = tensor_dict[values_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.axis = axis;
 

        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/onehot.spv")).c_str());
        program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
        program->spec(64, 64, 64);
        //program->bind(binding, *tensor_dict[indices_input]->data(), *tensor_dict[depth_input]->data(), *tensor_dict[values_input]->data(), *tensor_dict[output_output]->data());
    }

}

